import os
import numpy as np
import xgboost as xgb
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, matthews_corrcoef

from omniphish.dataset_loader import PhishingDataset, custom_collate
from omniphish.cnn_model import CNN1DEmbedding
from omniphish.transformer_model import CodeBERTEmbedding

def extract_global_features():
    print("[*] Loading trained Neural Networks...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    cnn = CNN1DEmbedding(embedding_dim=64, num_filters=128).to(device)
    try:
        cnn.load_state_dict(torch.load("weights/cnn_trained.pt", map_location=device))
    except Exception as e:
        print(f"[!] Failed to load CNN weights. Please run trainer.py first! Error: {e}")
        return None, None, None, None
    cnn.eval()
    
    codebert = CodeBERTEmbedding(use_lora=True).to(device)
    try:
        codebert.load_state_dict(torch.load("weights/codebert_trained.pt", map_location=device))
    except Exception as e:
        print(f"[!] Failed to load CodeBERT weights. Error: {e}")
        return None, None, None, None
    codebert.eval()
    
    dirs = {'phishing': 'dataset/raw_html/phishing', 'benign': 'dataset/raw_html/benign'}
    dataset = PhishingDataset(dirs, undersample_benign=False)
    
    if len(dataset) == 0:
        print("[!] Dataset is empty.")
        return None, None, None, None
        
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=custom_collate)
    
    cnn_feats, cb_feats, heuristics, labels_list = [], [], [], []
    
    print(f"[*] Dynamically Extracting Deep Learning features for {len(dataset)} files...")
    with torch.no_grad():
        for batch_dicts, labels in tqdm(dataloader, desc="Ablation Extraction"):
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
    
    if len(heuristics.shape) == 1:
        heuristics = heuristics.reshape(-1, 1)
        
    return cnn_feats, cb_feats, heuristics, y_true

def run_ablation_test(test_name, X, y, test_idx):
    print("="*60)
    print(f"🔬 RUNNING ABLATION: {test_name}")
    print(f"[*] Feature Space Dimensions: {X.shape[1]}-D")
    
    # 1. Isolate the Zero-Day Vault
    indices = np.arange(len(y))
    train_val_idx = np.setdiff1d(indices, test_idx)
    
    X_train_raw = X[train_val_idx]
    y_train_raw = y[train_val_idx]
    
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    # 2. Fix the Imbalanced Dataset using SMOTE!
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_raw, y_train_raw)
    
    # 3. Train the Meta-Classifier
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_smote, y_train_smote)
    
    # 4. Evaluate on Zero-Day Vault
    all_preds = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else np.zeros_like(all_preds, dtype=float)
    
    acc = accuracy_score(y_test, all_preds)
    prec = precision_score(y_test, all_preds, zero_division=0)
    rec = recall_score(y_test, all_preds, zero_division=0)
    f1 = f1_score(y_test, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_probs)
    except:
        auc = 0.0
    mcc = matthews_corrcoef(y_test, all_preds)
    
    print(f"    --> F1-Score:  {f1*100:.2f}%")
    print(f"    --> Precision: {prec*100:.2f}%")
    print(f"    --> Recall:    {rec*100:.2f}%")
    print(f"    --> ROC-AUC:   {auc:.4f}")
    print(f"    --> MCC:       {mcc:.4f}")
    print("="*60 + "\n")
    
    return {
        "Test": test_name,
        "F1": f1 * 100,
        "Precision": prec * 100,
        "Recall": rec * 100,
        "MCC": mcc
    }

def main():
    print("🚀 INITIALIZING ABLATION STUDY...")
    
    if not os.path.exists("weights/vault_indices.npy"):
        print("[!] Zero-Day Vault indices not found! Please run trainer.py first.")
        return
        
    test_idx = np.load("weights/vault_indices.npy")
    
    # Dynamically extract features since trainer.py doesn't cache them globally
    X_cnn, X_codebert, X_heuristics, y_all = extract_global_features()
    if X_cnn is None:
        return
        
    print("\n[+] All vectors extracted successfully. Beginning mathematically isolated tests...\n")
    
    results = []
    
    # Test 1: No CodeBERT (CNN + Heuristics)
    X_test1 = np.hstack((X_cnn, X_heuristics))
    results.append(run_ablation_test("No CodeBERT (CNN + Heuristics)", X_test1, y_all, test_idx))
    
    # Test 2: No CNN (CodeBERT + Heuristics)
    X_test2 = np.hstack((X_codebert, X_heuristics))
    results.append(run_ablation_test("No CNN (CodeBERT + Heuristics)", X_test2, y_all, test_idx))
    
    # Test 3: No Heuristics (CodeBERT + CNN)
    X_test3 = np.hstack((X_cnn, X_codebert))
    results.append(run_ablation_test("No Heuristics (CodeBERT + CNN)", X_test3, y_all, test_idx))
    
    # Test 4: Heuristics Only
    X_test4 = X_heuristics
    results.append(run_ablation_test("Heuristics Only (No Deep Learning)", X_test4, y_all, test_idx))
    
    # Test 5: Full Tri-Modal Ensemble
    X_test5 = np.hstack((X_cnn, X_codebert, X_heuristics))
    results.append(run_ablation_test("FULL ENSEMBLE (All 903 Dimensions)", X_test5, y_all, test_idx))
    
    # Print Final Summary Table
    print("\n" + "="*85)
    print(f"{'🏆 FINAL ABLATION STUDY SUMMARY 🏆':^85}")
    print("="*85)
    print(f"{'Ablation Test':<40} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'MCC':<10}")
    print("-" * 85)
    for res in results:
        print(f"{res['Test']:<40} | {res['F1']:.2f}%     | {res['Precision']:.2f}%     | {res['Recall']:.2f}%     | {res['MCC']:.4f}")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
