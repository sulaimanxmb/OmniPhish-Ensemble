import os
import torch
import numpy as np
from bs4 import BeautifulSoup
from torch.utils.data import Dataset, DataLoader
from .html_parser import clean_html, extract_codebert_tags
from .url_heuristics import is_suspicious_action

def extract_dom_graph(soup, max_nodes=1024):
    """
    Parses the BeautifulSoup DOM tree into a graph structure.
    Returns:
        nodes: Tensor of shape (max_nodes,) containing tag IDs
        adj: Tensor of shape (max_nodes, max_nodes) representing adjacency
    """
    nodes = []
    edges = []
    
    def traverse(element, parent_idx=-1):
        if len(nodes) >= max_nodes:
            return
            
        current_idx = len(nodes)
        
        tag_name = element.name if element.name else "text"
        # simple hash for tag
        tag_id = (hash(tag_name) % 255) + 1
        nodes.append(tag_id)
        
        if parent_idx != -1:
            edges.append((parent_idx, current_idx))
            edges.append((current_idx, parent_idx)) # undirected graph for better info flow
            
        if hasattr(element, 'children'):
            for child in element.children:
                if child.name is not None:
                    traverse(child, current_idx)
                    
    body = soup.find('body')
    root = body if body else soup
    traverse(root, -1)
    
    num_nodes = len(nodes)
    if num_nodes < max_nodes:
        nodes.extend([0] * (max_nodes - num_nodes))
        
    adj = torch.zeros((max_nodes, max_nodes), dtype=torch.float32)
    for u, v in edges:
        if u < max_nodes and v < max_nodes:
            adj[u, v] = 1.0
            
    return torch.tensor(nodes, dtype=torch.long), adj

def get_dom_depth_stats(soup):
    """
    Recursively calculates the Maximum and Average DOM tree depth.
    Phishing kits often use heavily obfuscated, deeply nested div tags.
    """
    max_depth = 0
    total_depth = 0
    node_count = 0
    
    def traverse(element, depth):
        nonlocal max_depth, total_depth, node_count
        if hasattr(element, 'children'):
            for child in element.children:
                if child.name is not None:  # Ensure it is an actual HTML Tag
                    node_count += 1
                    total_depth += depth
                    if depth > max_depth:
                        max_depth = depth
                    traverse(child, depth + 1)
                    
    # Focus on the body to ignore head/meta data noise if possible
    body = soup.find('body')
    root = body if body else soup
    traverse(root, 1)
    
    avg_depth = total_depth / node_count if node_count > 0 else 0.0
    return float(max_depth), float(avg_depth)

class PhishingDataset(Dataset):
    def __init__(self, raw_html_dirs, undersample_benign=False):
        """
        raw_html_dirs: Dict -> {'phishing': 'path', 'benign': 'path'}
        undersample_benign: If True, specifically truncates benign data to match phishing natively in RAM.
        """
        import random
        
        phish_samples = []
        benign_samples = []
        
        for class_label, folder_path in raw_html_dirs.items():
            # Resolve absolute path to prevent Windows Multiprocessing Errno 22 crashes
            folder_path = os.path.abspath(folder_path)
            if not os.path.exists(folder_path):
                continue
            
            files = [f for f in os.listdir(folder_path) if f.endswith(".html")]
            
            # No artificial capping applied; utilizing the full Phishing dataset.
                
            for filename in files:
                # Store absolute normalized paths
                filepath = os.path.normpath(os.path.join(folder_path, filename))
                if class_label == 'phishing':
                    phish_samples.append(filepath)
                else:
                    benign_samples.append(filepath)
                        
        if undersample_benign and len(phish_samples) < len(benign_samples):
            print(f"\n[PyTorch Pre-Loader] Slicing Benign memory footprint from {len(benign_samples)} down to {len(phish_samples)} to maintain symmetry!")
            random.shuffle(benign_samples)
            benign_samples = benign_samples[:len(phish_samples)]
            
        self.samples = phish_samples + benign_samples
        self.labels = [1] * len(phish_samples) + [0] * len(benign_samples)
        
        # Formulate and cleanly shuffle the exact sequences for PyTorch to consume
        combined = list(zip(self.samples, self.labels))
        random.shuffle(combined)
        if combined:
            self.samples, self.labels = zip(*combined)
        self.samples = list(self.samples)
        self.labels = list(self.labels)
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath = self.samples[idx]
        label = self.labels[idx]
        
        # Aggressive cleaning for Windows Multiprocessing string IPC corruption
        if isinstance(filepath, str):
            filepath = filepath.replace('\x00', '').strip()
            if os.name == 'nt' and not filepath.startswith('\\\\?\\'):
                # Bypass Win32 MAX_PATH and strict character validation limits
                filepath = '\\\\?\\' + filepath
            
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw_html = f.read()
        except OSError as e:
            print(f"\n[!] WARNING: Windows IPC failed to read {filepath} - injecting blank fallback.")
            raw_html = "<html></html>"
            
        # SAFETY: Obfuscated phishing files can sometimes be 10+ MB of Base64 on a single line.
        # BeautifulSoup will completely freeze your CPU for 60+ seconds trying to build a DOM for this.
        # We aggressively truncate to the first 100,000 characters to prevent pipeline stalls.
        raw_html = raw_html[:100000]
            
        cleaned_html = clean_html(raw_html)
        
        # 1. Provide Graph representation for GNN
        soup = BeautifulSoup(raw_html, 'html.parser')
        gnn_nodes, gnn_adj = extract_dom_graph(soup, max_nodes=1024)
        
        # Extract tags for CodeBERT directly as string
        codebert_text = extract_codebert_tags(cleaned_html)
        
        # Extract Suspicious Form Action and DOM Depth heuristic
        soup = BeautifulSoup(raw_html, 'html.parser')
        suspicious_form_action = 0.0
        for form in soup.find_all('form'):
            if is_suspicious_action(form.get('action', '')):
                suspicious_form_action = 1.0
                break
                
        max_depth, avg_depth = get_dom_depth_stats(soup)
                
        return {
            'gnn_nodes': gnn_nodes,
            'gnn_adj': gnn_adj,
            'codebert_text': codebert_text,
            'heuristic': torch.tensor([suspicious_form_action, max_depth, avg_depth], dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

def custom_collate(batch):
    """
    Since graphs have varying nodes, we can't easily batch adj matrices statically.
    For simplicity for our sequential architecture, we'll process batch_sizes
    by keeping them inside a list and stacking the 1D vectors downstream.
    """
    labels = torch.stack([item['label'] for item in batch])
    return batch, labels

if __name__ == "__main__":
    dirs = {
        'phishing': 'dataset/raw_html/phishing',
        'benign': 'dataset/raw_html/benign'
    }
    dataset = PhishingDataset(dirs)
    print(f"Dataset length: {len(dataset)}")
    dl = DataLoader(dataset, batch_size=2, collate_fn=custom_collate)
    for b, lbls in dl:
        print("Batch size:", len(b))
        print("Labels:", lbls)
        break
