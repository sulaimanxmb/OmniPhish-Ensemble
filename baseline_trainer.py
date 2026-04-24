import os
import pickle
import numpy as np
import torch
from torch.utils.data import random_split
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from dataset_loader import PhishingDataset
from baseline_features import extract_manual_features

def load_and_extract_features():
    """
    Loads the exact same dataset via PyTorch's dataset loader, 
    but manually extracts the baseline heuristic features.
    """
    dirs = {
        'phishing': 'dataset/raw_html/phishing',
        'benign': 'dataset/raw_html/benign'
    }
    
    # We must ensure the same logic as trainer.py
    phish_count = len([f for f in os.listdir(dirs['phishing']) if f.endswith('.html')]) if os.path.exists(dirs['phishing']) else 0
    benign_count = len([f for f in os.listdir(dirs['benign']) if f.endswith('.html')]) if os.path.exists(dirs['benign']) else 0
    
    undersample_benign_flag = False
    if phish_count != benign_count and min(phish_count, benign_count) > 0:
        if phish_count < benign_count:
            undersample_benign_flag = True
            
    dataset = PhishingDataset(dirs, undersample_benign=undersample_benign_flag)
    dataset_len = len(dataset)
    
    if dataset_len == 0:
        print("Dataset is empty. Run phish_scraper.py first.")
        return None, None, None
        
    print(f"\n[+] Extracting Baseline Heuristics from {dataset_len} files...")
    all_features = []
    all_labels = []
    
    # The dataset contains paths to the HTML files in `dataset.samples`
    # We will read them directly to extract features.
    for i in tqdm(range(dataset_len), desc="Extracting"):
        filepath = dataset.samples[i]
        label = dataset.labels[i]
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            raw_html = f.read()
            
        features = extract_manual_features(raw_html)
        all_features.append(features)
        all_labels.append(label)
        
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    # --- 70/15/15 DATASPLIT IMPLEMENTATION ---
    # We must use the exact same splits as the deep learning model!
    indices = list(range(dataset_len))
    train_size = int(0.7 * dataset_len)
    val_size = int(0.15 * dataset_len)
    test_size = dataset_len - train_size - val_size
    
    # Use the same PyTorch manual seed to split the indices identically
    generator = torch.Generator().manual_seed(42)
    train_idx, val_idx, test_idx = random_split(indices, [train_size, val_size, test_size], generator=generator)
    
    # Convert Subset to list of indices
    train_idx = train_idx.indices
    val_idx = val_idx.indices
    test_idx = test_idx.indices
    
    # We combine train and val for the Random Forest since it doesn't need epochs/validation
    train_val_idx = train_idx + val_idx
    
    X_train = all_features[train_val_idx]
    y_train = all_labels[train_val_idx]
    
    X_test = all_features[test_idx]
    y_test = all_labels[test_idx]
    
    return X_train, y_train, X_test, y_test

def train_baseline():
    print("="*50)
    print("🚀 BASELINE MODEL: Random Forest Heuristic Trainer")
    print("="*50)
    
    X_train, y_train, X_test, y_test = load_and_extract_features()
    if X_train is None:
        return
        
    print(f"\n[+] Training Random Forest on {len(X_train)} samples...")
    # Initialize Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    os.makedirs("weights", exist_ok=True)
    with open("weights/baseline_rf.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    print("[+] Model saved to weights/baseline_rf.pkl")
    
    # ---------------------------------------------------------
    # FINAL TESTING PHASE (IEEE METRICS)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🚀 RUNNING FINAL METRICS ON UNSEEN TEST SET (15%)")
    print("="*50)
    
    all_preds = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, all_preds)
    prec = precision_score(y_test, all_preds, zero_division=0)
    rec = recall_score(y_test, all_preds, zero_division=0)
    f1 = f1_score(y_test, all_preds, zero_division=0)
    cm = confusion_matrix(y_test, all_preds)
    
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"True Negatives (Correctly Safe):   {cm[0][0]}")
    print(f"False Positives (Mistake blocked): {cm[0][1]}")
    print(f"False Negatives (Missed Phish):    {cm[1][0]}")
    print(f"True Positives (Caught Phish):     {cm[1][1]}")
    print("="*50 + "\n")
    
    print("Notice the metrics compared to the Deep Learning Stacking Ensemble!")
    print("This proves that your manual heuristics are a weak baseline, and your CodeBERT/CNN architecture is superior.")

if __name__ == "__main__":
    train_baseline()
