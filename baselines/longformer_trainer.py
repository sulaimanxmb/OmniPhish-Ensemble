import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from omniphish.dataset_loader import PhishingDataset, custom_collate
from transformers import AutoTokenizer, LongformerForSequenceClassification
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

def train_longformer():
    print("\n" + "="*50)
    print("🤖 RUNNING BASELINE: Longformer (Long Context 4096)")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on: {device}")
    
    dirs = {
        'phishing': 'dataset/raw_html/phishing',
        'benign': 'dataset/raw_html/benign'
    }
    
    # Load dataset
    full_dataset = PhishingDataset(dirs, undersample_benign=True)
    
    # 10% Zero-Day Vault split
    vault_size = int(len(full_dataset) * 0.10)
    train_val_size = len(full_dataset) - vault_size
    train_val_dataset, vault_dataset = torch.utils.data.random_split(
        full_dataset, [train_val_size, vault_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Small batch size to accommodate 4096 tokens
    train_loader = DataLoader(train_val_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate, num_workers=0)
    vault_loader = DataLoader(vault_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate, num_workers=0)
    
    model_name = "allenai/longformer-base-4096"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load Longformer for sequence classification
    model = LongformerForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    
    # Longformer uses BCE loss with logits for binary classification natively if num_labels=1
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    accumulation_steps = 16 # Effective batch size 32
    
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for i, (batch, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Get raw text
            texts = [item['codebert_text'] for item in batch]
            labels = labels.unsqueeze(1).to(device, dtype=torch.float)
            
            # Tokenize up to 4096 tokens natively
            encodings = tokenizer(texts, truncation=True, padding=True, max_length=4096, return_tensors="pt")
            
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            # Use AMP (Mixed Precision) to save VRAM
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                # Normalize loss for gradient accumulation
                loss = loss / accumulation_steps
                
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            total_loss += loss.item() * accumulation_steps
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        
    print("\n[*] Evaluating on Zero-Day Vault...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch, labels in tqdm(vault_loader, desc="Evaluating"):
            texts = [item['codebert_text'] for item in batch]
            encodings = tokenizer(texts, truncation=True, padding=True, max_length=4096, return_tensors="pt")
            
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
            preds = torch.sigmoid(logits).squeeze().cpu().numpy()
            
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
    prec = precision_score(all_labels, pred_classes, zero_division=0)
    rec = recall_score(all_labels, pred_classes, zero_division=0)
    f1 = f1_score(all_labels, pred_classes, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.0
    
    print("\n==================================================")
    print("📊 LONGFORMER ZERO-DAY VAULT METRICS")
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
    train_longformer()
