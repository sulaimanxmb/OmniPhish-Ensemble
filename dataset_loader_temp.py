import os
import random
from dataset_loader import PhishingDataset as OriginalDataset
from html_parser import clean_html, extract_codebert_tags
from cnn_model import text_to_tensor
from baseline_features import is_suspicious_action
from dataset_loader import get_dom_depth_stats
from bs4 import BeautifulSoup
import torch

class PhishingDatasetTemp(OriginalDataset):
    def __init__(self, raw_html_dirs, undersample_benign=False):
        phish_samples = []
        benign_samples = []
        
        for class_label, folder_path in raw_html_dirs.items():
            if not os.path.exists(folder_path):
                continue
            files = [f for f in os.listdir(folder_path) if f.endswith(".html")]
            if class_label == 'phishing' and len(files) > 2000:
                print(f"\n[TEMP] Randomly selecting 2000 Phishing samples from {len(files)}.")
                random.seed(42) # Deterministic random
                random.shuffle(files)
                files = files[:2000]
            for filename in files:
                filepath = os.path.join(folder_path, filename)
                if class_label == 'phishing':
                    phish_samples.append(filepath)
                else:
                    benign_samples.append(filepath)
                        
        self.samples = phish_samples + benign_samples
        self.labels = [1] * len(phish_samples) + [0] * len(benign_samples)
        
        combined = list(zip(self.samples, self.labels))
        random.seed(42)
        random.shuffle(combined)
        if combined:
            self.samples, self.labels = zip(*combined)
        self.samples = list(self.samples)
        self.labels = list(self.labels)
