import os
import torch
import numpy as np
from bs4 import BeautifulSoup
from torch.utils.data import Dataset, DataLoader
from html_parser import clean_html, extract_codebert_tags
from cnn_model import text_to_tensor
from baseline_features import is_suspicious_action

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
            if not os.path.exists(folder_path):
                continue
            
            files = [f for f in os.listdir(folder_path) if f.endswith(".html")]
            
            # No artificial capping applied; utilizing the full Phishing dataset.
                
            for filename in files:
                filepath = os.path.join(folder_path, filename)
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
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            raw_html = f.read()
            
        cleaned_html = clean_html(raw_html)
        
        # 1. Provide ASCII/Byte representation for CNN
        # Shape output from text_to_tensor is (1, seq_len) -> we squeeze to (seq_len)
        cnn_tensor = text_to_tensor(cleaned_html, max_len=1024).squeeze(0)
        
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
            'cnn_input': cnn_tensor,
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
