import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, matthews_corrcoef
import random

from omniphish.dataset_loader import PhishingDataset

def set_seed(seed=42):
    """Locks all random seeds for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# Hyperparameters
VOCAB_SIZE = 5000
MAX_LEN = 2000
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001

class RawHTMLCNN(nn.Module):
    def __init__(self):
        super(RawHTMLCNN, self).__init__()
        # Blindly converts raw hashed words into 64-D vectors
        self.embedding = nn.Embedding(VOCAB_SIZE, 64)
        # Scans for raw structural patterns without semantic understanding
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch, max_len)
        x = self.embedding(x) # (batch, max_len, 64)
        x = x.transpose(1, 2) # (batch, 64, max_len)
        x = self.conv1(x)     # (batch, 128, max_len)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1) # (batch, 128)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def preprocess_raw_html():
    dirs = {
        'phishing': 'dataset/raw_html/phishing',
        'benign': 'dataset/raw_html/benign'
    }
    
    dataset = PhishingDataset(dirs, undersample_benign=False)
    dataset_len = len(dataset)
    
    if dataset_len == 0:
        print("[!] Dataset is empty.")
        return None, None, None, None
        
    print(f"\n[+] Processing {dataset_len} raw HTML files for SOTA-2 CNN...")
    
    all_sequences = []
    all_labels = []
    
    # Fast Hashing Tokenizer (Simulating a raw, blind tokenizer used in early Deep Learning papers)
    for i in tqdm(range(dataset_len), desc="Blindly Tokenizing HTML"):
        filepath = dataset.samples[i]
        label = dataset.labels[i]
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw_html = f.read()
        except:
            raw_html = ""
            
        # Treat raw HTML simply as space-separated words, separating brackets to catch tags
        words = raw_html.replace('>', ' > ').replace('<', ' < ').split()
        
        # Hash words to integer indices (0 to 4999)
        seq = [hash(w) % VOCAB_SIZE for w in words[:MAX_LEN]]
        
        # Pad sequence if it's too short
        if len(seq) < MAX_LEN:
            seq = seq + [0] * (MAX_LEN - len(seq))
            
        all_sequences.append(seq)
        all_labels.append(label)
        
    X = torch.tensor(all_sequences, dtype=torch.long)
    y = torch.tensor(all_labels, dtype=torch.float32)
    
    # --- EXACT ZERO-DAY VAULT ISOLATION ---
    indices = np.arange(dataset_len)
    if os.path.exists("weights/vault_indices.npy"):
        print("[+] Loading exact Zero-Day Vault indices to ensure 1:1 paper comparison...")
        test_idx = np.load("weights/vault_indices.npy")
        train_val_idx = np.setdiff1d(indices, test_idx)
    else:
        print("[!] No vault_indices.npy found. Falling back to random split.")
        from sklearn.model_selection import train_test_split
        train_val_idx, test_idx = train_test_split(indices, test_size=0.1, stratify=y.numpy(), random_state=42)
        
    X_train = X[train_val_idx]
    y_train = y[train_val_idx]
    
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    return X_train, y_train, X_test, y_test

def train_sota2():
    print("="*70)
    print("🚀 SOTA-2 MODEL: Raw HTML 1D-CNN (HTMLPhish Architecture)")
    print("="*70)
    
    X_train, y_train, X_test, y_test = preprocess_raw_html()
    if X_train is None:
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[*] Training on device: {device}")
    
    # Cost-Sensitive Learning: Mathematically weight the Benign class to fix the 1:4.5 imbalance
    num_phishing = y_train.sum().item()
    num_benign = len(y_train) - num_phishing
    weight_for_phishing = num_benign / num_phishing if num_phishing > 0 else 1.0
    
    pos_weight = torch.tensor([weight_for_phishing]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    model = RawHTMLCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4 if os.name == 'nt' else 8, pin_memory=True, persistent_workers=True)
    
    print(f"\n[+] Training Raw CNN for {EPOCHS} Epochs...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"    --> Loss: {epoch_loss / len(train_loader):.4f}")
        
    print("\n" + "="*70)
    print("🚀 RUNNING FINAL METRICS ON EXACT ZERO-DAY VAULT (10%)")
    print("="*70)
    
    model.eval()
    X_test_mps = X_test.to(device)
    with torch.no_grad():
        logits = model(X_test_mps).squeeze(-1)
        y_probs = torch.sigmoid(logits).cpu().numpy()
        all_preds = (y_probs > 0.5).astype(int)
        
    y_test_np = y_test.numpy()
    
    acc = accuracy_score(y_test_np, all_preds)
    prec = precision_score(y_test_np, all_preds, zero_division=0)
    rec = recall_score(y_test_np, all_preds, zero_division=0)
    f1 = f1_score(y_test_np, all_preds, zero_division=0)
    cm = confusion_matrix(y_test_np, all_preds)
    
    auc = roc_auc_score(y_test_np, y_probs)
    mcc = matthews_corrcoef(y_test_np, all_preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"FPR:       {fpr*100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"True Negatives (Correctly Safe):   {cm[0][0]}")
    print(f"False Positives (Mistake blocked): {cm[0][1]}")
    print(f"False Negatives (Missed Phish):    {cm[1][0]}")
    print(f"True Positives (Caught Phish):     {cm[1][1]}")
    print("="*70 + "\n")

if __name__ == "__main__":
    train_sota2()
