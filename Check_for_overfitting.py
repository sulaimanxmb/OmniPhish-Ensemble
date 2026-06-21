import os
import numpy as np
import pandas as pd
import torch
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from omniphish.dataset_loader import PhishingDataset, custom_collate
from omniphish.cnn_model import CNN1DEmbedding
from omniphish.transformer_model import CodeBERTEmbedding
from omniphish.classifier import MetaClassifier

def analyze_kfold_variance():
    print("\n" + "="*60)
    print("📊 PART 1: K-FOLD VARIANCE ANALYSIS (METHOD 2)")
    print("="*60)
    
    if not os.path.exists("metrics/kfold_variance_logs.csv"):
        print("[!] Error: 'metrics/kfold_variance_logs.csv' not found. Run trainer.py first.")
        return False
        
    df = pd.read_csv("metrics/kfold_variance_logs.csv")
    
    mean_train_f1 = df["Train_F1"].mean() * 100
    mean_val_f1 = df["Val_F1"].mean() * 100
    f1_variance_drop = mean_train_f1 - mean_val_f1
    
    mean_train_prec = df["Train_Precision"].mean() * 100
    mean_val_prec = df["Val_Precision"].mean() * 100
    prec_variance_drop = mean_train_prec - mean_val_prec
    
    print(f"Average Training F1:     {mean_train_f1:.2f}%")
    print(f"Average Validation F1:   {mean_val_f1:.2f}%")
    print(f"--> F1 Variance Drop:    {f1_variance_drop:.2f}%")
    print("-" * 30)
    print(f"Average Training Prec:   {mean_train_prec:.2f}%")
    print(f"Average Validation Prec: {mean_val_prec:.2f}%")
    print(f"--> Prec Variance Drop:  {prec_variance_drop:.2f}%")
    
    print("\n[🤖 AI ASSESSMENT]")
    if f1_variance_drop < 5.0:
        print("✅ EXCELLENT: Model exhibits very low variance (<5%). It is highly generalized and NOT overfitting to the training split.")
    elif f1_variance_drop < 10.0:
        print("⚠️ ACCEPTABLE: Moderate variance. The model is slightly memorizing the training data, but generalization is still solid.")
    else:
        print("❌ DANGER: High variance (>10%). The model has severely overfitted to the training data and will likely fail in the real world.")
        
    return True

def test_zero_day_vault():
    print("\n" + "="*60)
    print("🛡️ PART 2: ZERO-DAY VAULT INFERENCE (METHOD 3)")
    print("="*60)
    
    if not os.path.exists("weights/vault_indices.npy"):
        print("[!] Error: 'weights/vault_indices.npy' not found. Ensure trainer.py isolated the vault.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Running inference on device: {device}")
    
    vault_idx = np.load("weights/vault_indices.npy")
    print(f"[*] Loaded {len(vault_idx)} isolated Zero-Day samples from the Vault.")
    
    dirs = {'phishing': 'dataset/raw_html/phishing', 'benign': 'dataset/raw_html/benign'}
    dataset = PhishingDataset(dirs, undersample_benign=False)
    vault_subset = Subset(dataset, vault_idx)
    vault_loader = DataLoader(vault_subset, batch_size=4, shuffle=False, collate_fn=custom_collate)
    
    # 1. Load Models
    print("[*] Loading trained Neural Networks and Meta-Classifier...")
    cnn = CNN1DEmbedding(embedding_dim=64, num_filters=128).to(device)
    try:
        cnn.load_state_dict(torch.load("weights/cnn_trained.pt", map_location=device))
    except Exception as e:
        print(f"[!] Failed to load CNN weights: {e}")
        return
    cnn.eval()
    
    codebert = CodeBERTEmbedding(use_lora=True).to(device)
    try:
        codebert.load_state_dict(torch.load("weights/codebert_trained.pt", map_location=device))
    except Exception as e:
        print(f"[!] Failed to load CodeBERT weights: {e}")
        return
    codebert.eval()
    
    meta = MetaClassifier()
    try:
        meta.load("weights/meta_classifier.pkl")
    except Exception as e:
        print(f"[!] Failed to load XGBoost Meta-Classifier: {e}")
        return
        
    # 2. Extract Features
    print("[*] Extracting Structural & Semantic features for Vault data...")
    cnn_feats, cb_feats, heuristics, labels_list = [], [], [], []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_dicts, labels in tqdm(vault_loader, desc="Vault Extraction"):
            for item, label in zip(batch_dicts, labels):
                c_emb = cnn(item['cnn_input'].unsqueeze(0).to(device)).cpu().numpy().flatten()
                cb_emb = codebert.compute_embedding(item['codebert_text']).cpu().numpy().flatten()
                
                cnn_feats.append(c_emb)
                cb_feats.append(cb_emb)
                heuristics.append(item['heuristic'].cpu().numpy())
                labels_list.append(label.item())
                
    cnn_feats = np.array(cnn_feats)
    cb_feats = np.array(cb_feats)
    heuristics = np.array(heuristics)
    y_true = np.array(labels_list)
    
    # Safe reshape for heuristics
    if len(heuristics.shape) == 1:
        heuristics = heuristics.reshape(-1, 1)
        
    X_vault = np.concatenate([cnn_feats, cb_feats, heuristics], axis=1)
    
    # 3. Predict & Evaluate
    print("[*] Running XGBoost Meta-Classifier on Vault features...")
    y_pred = [meta.predict(x) for x in X_vault]
    
    end_time = time.time()
    
    # Calculate Latency & RAM
    total_time_seconds = end_time - start_time
    total_samples = len(vault_idx)
    ms_per_url = (total_time_seconds / total_samples) * 1000
    process = psutil.Process(os.getpid())
    peak_ram_mb = process.memory_info().rss / (1024 * 1024)
    
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    
    print("\n" + "="*40)
    print("🏆 FINAL ZERO-DAY VAULT METRICS 🏆")
    print("="*40)
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall:    {rec:.2f}%")
    print(f"F1-Score:  {f1:.2f}%")
    print("-" * 40)
    print(f"Latency:   {ms_per_url:.2f} ms per URL")
    print(f"Peak RAM:  {peak_ram_mb:.2f} MB")
    print("="*40)
    
    print("\n[Detailed Classification Report]")
    print(classification_report(y_true, y_pred, target_names=["Benign (0)", "Phishing (1)"]))
    
    # 4. Generate Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Benign', 'Predicted Phishing'],
                yticklabels=['Actual Benign', 'Actual Phishing'])
    plt.title('Zero-Day Vault Confusion Matrix')
    plt.tight_layout()
    plt.savefig('vault_confusion_matrix.png', dpi=300)
    print("\n[+] Saved 'vault_confusion_matrix.png' for IEEE paper.")

if __name__ == "__main__":
    if analyze_kfold_variance():
        test_zero_day_vault()
