import os
import torch
import numpy as np
from bs4 import BeautifulSoup
from torch.utils.data import Dataset, DataLoader
from html_parser import clean_html, extract_codebert_tags
from cnn_model import text_to_tensor
from baseline_features import is_suspicious_action

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
            
            # Cap phishing samples to 700 maximum as requested
            if class_label == 'phishing' and len(files) > 700:
                print(f"\n[PyTorch Pre-Loader] Capping Phishing dataset from {len(files)} down to the FIRST 700 samples for consistent stats.")
                files.sort()  # Sort alphabetically to guarantee the exact same 700 files are chosen every time
                files = files[:700]
                
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
        
        # Extract Suspicious Form Action heuristic
        soup = BeautifulSoup(raw_html, 'html.parser')
        suspicious_form_action = 0
        for form in soup.find_all('form'):
            if is_suspicious_action(form.get('action', '')):
                suspicious_form_action = 1
                break
                
        return {
            'cnn_input': cnn_tensor,
            'codebert_text': codebert_text,
            'heuristic': torch.tensor([suspicious_form_action], dtype=torch.float32),
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
