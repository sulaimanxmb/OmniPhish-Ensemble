import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset_loader import PhishingDataset, custom_collate
from cnn_model import CNN1DEmbedding
from transformer_model import CodeBERTEmbedding
from classifier import MetaClassifier
import os
import random
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import numpy as np

def set_seed(seed=42):
    """Locks all random seeds for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Also for Apple MPS if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

def extract_features(cnn, codebert, dataloader, device):
    """
    Passes the dataset through the trained CNN and frozen CodeBERT
    to extract the final 1D feature vectors for XGBoost.
    """
    cnn.eval()
    all_features = []
    all_labels = []
    
    print(f"\n[+] Extracting Phase 2 Features (CNN + CodeBERT)...")
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Extracting Features", unit="batch")
        for batch_dicts, labels in progress_bar:
            for item, label in zip(batch_dicts, labels):
                cnn_input = item['cnn_input'].unsqueeze(0).to(device)
                cb_text = item['codebert_text']
                
                # Extract CNN features
                cnn_emb = cnn(cnn_input)
                
                # Extract CodeBERT features
                cb_emb = codebert.compute_embedding(cb_text)
                
                # Extract Heuristic feature
                heuristic_val = item['heuristic'].cpu().numpy()
                
                # Concatenate locally to save RAM later
                cnn_flat = cnn_emb.cpu().numpy().flatten()
                cb_flat = cb_emb.cpu().numpy().flatten()
                
                # Using the classifier's expected vector composition manually here
                combined = np.concatenate([cnn_flat, cb_flat, heuristic_val])
                all_features.append(combined)
                all_labels.append(label.item())
                
    return np.array(all_features), np.array(all_labels)

def train_model(epochs=5, batch_size=4):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # ---------------------------------------------------------
    # DATASET INITIALIZATION
    # ---------------------------------------------------------
    dirs = {
        'phishing': 'dataset/raw_html/phishing',
        'benign': 'dataset/raw_html/benign'
    }
    
    phish_count = len([f for f in os.listdir(dirs['phishing']) if f.endswith('.html')]) if os.path.exists(dirs['phishing']) else 0
    benign_count = len([f for f in os.listdir(dirs['benign']) if f.endswith('.html')]) if os.path.exists(dirs['benign']) else 0
    
    undersample_benign_flag = False
    if phish_count != benign_count and min(phish_count, benign_count) > 0:
        if phish_count < benign_count:
            print("\n[!] Minority class is Phishing. Triggering PyTorch Undersampling on Benign domains to prevent XGBoost bias...")
            undersample_benign_flag = True
            
    dataset = PhishingDataset(dirs, undersample_benign=undersample_benign_flag)
    dataset_len = len(dataset)
    if dataset_len == 0:
        print("Dataset is empty. Ensure phish_scraper.py has run successfully and generated HTMLs.")
        return
        
    # --- 70/15/15 DATASPLIT IMPLEMENTATION ---
    train_size = int(0.7 * dataset_len)
    val_size = int(0.15 * dataset_len)
    test_size = dataset_len - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # Reproducibility for academic paper
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    
    print(f"\n[+] Dataset properly split for IEEE evaluation:")
    print(f"  - Total Files: {dataset_len}")
    print(f"  - Training Set (70%): {train_size} files")
    print(f"  - Validation Set (15%): {val_size} files")
    print(f"  - Test Set (15%): {test_size} files\n")
    
    # ---------------------------------------------------------
    # PHASE 1: TRAIN CNN REPRESENTATION
    # ---------------------------------------------------------
    print("="*50)
    print("🚀 PHASE 1: Training CNN Structural Extractor")
    print("="*50)
    cnn = CNN1DEmbedding().to(device)
    codebert = CodeBERTEmbedding().to(device)
    
    # We use a temporary Linear layer just to backprop gradients into the CNN
    temp_classifier = nn.Linear(128, 1).to(device) 
    
    optimizer = optim.Adam(list(cnn.parameters()) + list(temp_classifier.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        cnn.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [CNN Train]", unit="batch")
        for batch_dicts, labels in progress_bar:
            labels = labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            
            batch_logits = []
            for item in batch_dicts:
                cnn_input = item['cnn_input'].unsqueeze(0).to(device)
                cnn_emb = cnn(cnn_input) 
                batch_logits.append(cnn_emb)
                
            batch_tensor = torch.cat(batch_logits, dim=0) 
            logits = temp_classifier(batch_tensor)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_dicts)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
    print("[+] CNN Training Complete. Saving Weights...")
    os.makedirs("weights", exist_ok=True)
    torch.save(cnn.state_dict(), "weights/cnn_trained.pt")
    
    # ---------------------------------------------------------
    # PHASE 2: EXTRACT FEATURES FOR XGBOOST
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🚀 PHASE 2: Extracting Hybrid Vectors (CNN + CodeBERT)")
    print("="*50)
    
    X_train, y_train = extract_features(cnn, codebert, train_loader, device)
    X_val, y_val = extract_features(cnn, codebert, val_loader, device)
    X_test, y_test = extract_features(cnn, codebert, test_loader, device)
    
    # ---------------------------------------------------------
    # PHASE 3: TRAIN XGBOOST + LOGISTIC REGRESSION
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🚀 PHASE 3: Training XGBoost Stacking Ensemble")
    print("="*50)
    
    meta_model = MetaClassifier(use_logistic_regression=True)
    
    # We combine train + val for XGBoost training because XGBoost handles its own validation
    # or we just train on train_dataset and validate on val_dataset
    meta_model.train(X_train, y_train)
    meta_model.save("weights/meta_classifier.pkl")
    
    # ---------------------------------------------------------
    # FINAL TESTING PHASE (IEEE METRICS)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🚀 RUNNING FINAL METRICS ON UNSEEN TEST SET (15%)")
    print("="*50)
    
    all_preds = []
    for x in X_test:
        pred = meta_model.predict(x)
        all_preds.append(pred)
        
    acc = accuracy_score(y_test, all_preds)
    prec = precision_score(y_test, all_preds, zero_division=0)
    rec = recall_score(y_test, all_preds, zero_division=0)
    f1 = f1_score(y_test, all_preds, zero_division=0)
    cm = confusion_matrix(y_test, all_preds)
    
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%  <-- (When it flags Phishing, it's right {prec*100:.2f}% of the time)")
    print(f"Recall:    {rec*100:.2f}%  <-- (Caught {rec*100:.2f}% of all actual Phishing attacks)")
    print(f"F1-Score:  {f1*100:.2f}%  <-- (Overall harmonic balance metric)")
    print("\nConfusion Matrix:")
    print(f"True Negatives (Correctly Safe):   {cm[0][0]}")
    print(f"False Positives (Mistake blocked): {cm[0][1]}")
    print(f"False Negatives (Missed Phish):    {cm[1][0]}")
    print(f"True Positives (Caught Phish):     {cm[1][1]}")
    print("="*50 + "\n")

if __name__ == "__main__":
    train_model(epochs=5, batch_size=4)
