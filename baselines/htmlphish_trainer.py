import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from omniphish.dataset_loader import PhishingDataset, custom_collate
from tqdm import tqdm

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# HTMLPhish Pure Character-Level CNN Architecture
class HTMLPhishCNN(nn.Module):
    def __init__(self, vocab_size=256, embedding_dim=64):
        super(HTMLPhishCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.fc1 = nn.Linear(256 * 253, 512) # Based on max_len=1024 -> pool -> pool
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # x is (batch_size, seq_len)
        x = self.embedding(x) # (batch, seq, embed_dim)
        x = x.transpose(1, 2) # (batch, embed_dim, seq)
        
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_htmlphish():
    print("\n" + "="*50)
    print("🤖 RUNNING BASELINE: HTMLPhish (Pure Char-Level CNN)")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on: {device}")
    
    dirs = {
        'phishing': 'dataset/raw_html/phishing',
        'benign': 'dataset/raw_html/benign'
    }
    
    full_dataset = PhishingDataset(dirs, undersample_benign=True)
    
    # 10% Zero-Day Vault split
    vault_size = int(len(full_dataset) * 0.10)
    train_val_size = len(full_dataset) - vault_size
    train_val_dataset, vault_dataset = torch.utils.data.random_split(
        full_dataset, [train_val_size, vault_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Multiprocessing on Windows requires __main__ guard, which we have.
    # Set num_workers=0 to prevent potential IPC issues during simple baseline run
    train_loader = DataLoader(train_val_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate, num_workers=0)
    vault_loader = DataLoader(vault_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate, num_workers=0)
    
    model = HTMLPhishCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            cnn_inputs = torch.stack([item['cnn_input'] for item in batch]).to(device)
            labels = labels.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(cnn_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        
    print("\n[*] Evaluating on Zero-Day Vault...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch, labels in vault_loader:
            cnn_inputs = torch.stack([item['cnn_input'] for item in batch]).to(device)
            outputs = model(cnn_inputs)
            preds = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            # Handle batch_size = 1 case
            if preds.ndim == 0:
                preds = [preds.item()]
            else:
                preds = preds.tolist()
                
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_classes = (all_preds > 0.5).astype(int)
    
    acc = accuracy_score(all_labels, pred_classes)
    prec = precision_score(all_labels, pred_classes)
    rec = recall_score(all_labels, pred_classes)
    f1 = f1_score(all_labels, pred_classes)
    auc = roc_auc_score(all_labels, all_preds)
    
    print("\n==================================================")
    print("📊 HTMLPhish ZERO-DAY VAULT METRICS")
    print("==================================================")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"ROC-AUC:   {auc:.4f}")
    
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak VRAM: {peak_vram:.2f} GB")
        
    print("==================================================\n")
    
if __name__ == "__main__":
    train_htmlphish()
