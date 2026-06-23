import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from omniphish.dataset_loader import PhishingDataset, custom_collate
from omniphish.cnn_model import CNN1DEmbedding
from omniphish.transformer_model import CodeBERTEmbedding
from omniphish.classifier import MetaClassifier
import os
import random
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, roc_auc_score, matthews_corrcoef
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import time

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def verify_cuda_installation():
    """Checks if an NVIDIA GPU exists but PyTorch was installed without CUDA."""
    if not torch.cuda.is_available():
        import subprocess
        try:
            # If nvidia-smi executes successfully, they have an NVIDIA GPU
            subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
            print("\n" + "!"*80)
            print("🚨 CRITICAL WARNING: NVIDIA GPU DETECTED BUT PYTORCH CANNOT USE IT! 🚨")
            print("It appears you installed the CPU-only version of PyTorch.")
            print("Your massive RTX GPU is sitting idle. To fix this, stop the script and run:")
            print("\n  pip uninstall torch torchvision torchaudio -y")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n")
            print("!"*80 + "\n")
            import time
            time.sleep(5) # Give them time to read it
        except Exception:
            # Not an NVIDIA system (e.g., Mac with MPS, or pure CPU), which is fine.
            pass

set_seed(42)
verify_cuda_installation()

def get_training_mode():
    print("\n" + "="*50)
    print("🚀 SELECT TRAINING MODE")
    print("="*50)
    print("[1] FAST MODE (Global CodeBERT PEFT, Feature-Leak accepted, ~15 mins)")
    print("[2] SLOW MODE (Strict K-Fold CodeBERT Isolation, Zero-Leak, ~2 hours)")
    print("="*50)
    try:
        # Fallback to fast if running non-interactively
        choice = input("Enter 1 or 2 [Default: 1]: ").strip()
    except EOFError:
        choice = "1"
    
    return "slow" if choice == "2" else "fast"

def train_codebert_lora(train_loader, codebert, device, epochs=1):
    """Fine-tunes CodeBERT specifically for phishing vocabulary using LoRA."""
    print("[*] Fine-tuning CodeBERT via LoRA...")
    codebert.train()
    
    # Temporary classification head for fine-tuning the static feature extractor
    temp_classifier = nn.Linear(768, 1).to(device)
    
    trainable_params = [p for p in codebert.parameters() if p.requires_grad] + list(temp_classifier.parameters())
    if not trainable_params:
        print("[!] No trainable parameters found. LoRA may be disabled. Skipping fine-tuning.")
        return
        
    optimizer = optim.AdamW(trainable_params, lr=5e-4) # Slightly higher LR for fast LoRA convergence
    criterion = nn.BCEWithLogitsLoss()
    
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_dicts, labels in tqdm(train_loader, desc=f"CodeBERT LoRA Epoch {epoch+1}/{epochs}", leave=False):
            labels = labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                batch_embs = []
                for item in batch_dicts:
                    # compute_embedding automatically enables gradients for the forward pass
                    emb = codebert.compute_embedding(item['codebert_text'])
                    batch_embs.append(emb)
                    
                if len(batch_embs) == 0:
                    continue
                    
                batch_embs_tensor = torch.stack(batch_embs)
                logits = temp_classifier(batch_embs_tensor)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
    # Explicitly clear massive memory artifacts from the GPU to prevent PCIe thrashing
    codebert.zero_grad(set_to_none=True)
    del optimizer
    del temp_classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def global_pre_extract_codebert_heuristics(dataloader, codebert, device):
    """Extracts CodeBERT and Heuristic features for the ENTIRE dataset ONCE (Fast Mode)."""
    codebert.eval()
    cb_feats, heuristics, labels_list = [], [], []
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for batch_dicts, labels in tqdm(dataloader, desc="Global Extraction: CodeBERT & Heuristics"):
                for item, label in zip(batch_dicts, labels):
                    cb_emb = codebert.compute_embedding(item['codebert_text']).cpu().numpy().flatten()
                    cb_feats.append(cb_emb)
                
                # Squeeze the tensor if needed and convert to numpy
                h_val = item['heuristic'].cpu().numpy()
                heuristics.append(h_val)
                labels_list.append(label.item())
                
    return np.array(cb_feats), np.array(heuristics), np.array(labels_list)

def extract_all_features(dataloader, cnn, codebert, device, desc):
    """Extracts all 3 modalities dynamically per-fold (Slow Mode)."""
    cnn.eval()
    codebert.eval()
    cnn_feats, cb_feats, heuristics, labels_list = [], [], [], []
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for batch_dicts, labels in tqdm(dataloader, desc=desc, leave=False):
                for item, label in zip(batch_dicts, labels):
                    c_emb = cnn(item['cnn_input'].unsqueeze(0).to(device)).cpu().numpy().flatten()
                    cb_emb = codebert.compute_embedding(item['codebert_text']).cpu().numpy().flatten()
                
                cnn_feats.append(c_emb)
                cb_feats.append(cb_emb)
                heuristics.append(item['heuristic'].cpu().numpy())
                labels_list.append(label.item())
                
    return np.array(cnn_feats), np.array(cb_feats), np.array(heuristics), np.array(labels_list)

def train_cnn(train_loader, device, epochs=5):
    cnn = CNN1DEmbedding(embedding_dim=64, num_filters=128).to(device)
    temp_classifier = nn.Linear(128, 1).to(device)
    optimizer = optim.Adam(list(cnn.parameters()) + list(temp_classifier.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    cnn.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_dicts, labels in tqdm(train_loader, desc=f"CNN Epoch {epoch+1}/{epochs}", leave=False):
            labels = labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                batch_logits = []
                for item in batch_dicts:
                    cnn_input = item['cnn_input'].unsqueeze(0).to(device)
                    batch_logits.append(cnn(cnn_input))
                loss = criterion(temp_classifier(torch.cat(batch_logits, dim=0)), labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
    cnn.zero_grad(set_to_none=True)
    del optimizer
    del temp_classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return cnn

def extract_cnn_features(dataloader, cnn, device, desc):
    cnn.eval()
    cnn_feats = []
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for batch_dicts, _ in tqdm(dataloader, desc=desc, leave=False):
                for item in batch_dicts:
                    cnn_input = item['cnn_input'].unsqueeze(0).to(device)
                    c_emb = cnn(cnn_input).cpu().numpy().flatten()
                    cnn_feats.append(c_emb)
    return np.array(cnn_feats)

def objective(trial, X_train, y_train, X_val, y_val):
    xgb_params = {
        'max_depth': trial.suggest_int('xgb_max_depth', 2, 6),
        'learning_rate': trial.suggest_float('xgb_lr', 1e-3, 0.3, log=True),
        'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
        'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
        'tree_method': 'hist',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    meta = MetaClassifier(use_logistic_regression=False, xgb_params=xgb_params)
    meta.train(X_train, y_train)
    preds = [meta.predict(x) for x in X_val]
    return f1_score(y_val, preds, zero_division=0)

def train_model(batch_size=32, n_optuna_trials=10, n_splits=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    import csv
    mode = get_training_mode()
    
    dirs = {'phishing': 'dataset/raw_html/phishing', 'benign': 'dataset/raw_html/benign'}
    dataset = PhishingDataset(dirs, undersample_benign=False)
    if len(dataset) == 0:
        print("Dataset is empty.")
        return
        
    # --- PHASE 0: ZERO-DAY VAULT SPLIT ---
    all_indices = np.arange(len(dataset))
    all_labels = np.array(dataset.labels)
    
    print("\n" + "="*50)
    print("🔒 PHASE 0: ISOLATING ZERO-DAY VAULT (10%)")
    print("="*50)
    
    train_val_idx, vault_idx = train_test_split(all_indices, test_size=0.1, stratify=all_labels, random_state=42)
    os.makedirs("weights", exist_ok=True)
    np.save("weights/vault_indices.npy", vault_idx)
    print(f"[*] Locked {len(vault_idx)} samples into the Zero-Day Vault.")
    print(f"[*] Remaining {len(train_val_idx)} samples will be used for K-Fold CV.")
    
    # We create a Subset of the dataset that excludes the vault
    train_val_dataset = Subset(dataset, train_val_idx)
    train_val_labels = all_labels[train_val_idx]
    
    # For global fast extraction, we only want to extract on the train_val dataset
    train_val_loader = DataLoader(train_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=4 if os.name == 'nt' else 8, pin_memory=True, persistent_workers=True)
    
    cb_feats, heuristics = None, None
    if mode == "fast":
        print("\n" + "="*50)
        print("🚀 PHASE 1: GLOBAL CODEBERT PEFT & PRE-EXTRACTION")
        print("="*50)
        codebert = CodeBERTEmbedding(use_lora=True).to(device)
        train_codebert_lora(DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=4 if os.name == 'nt' else 8, pin_memory=True, persistent_workers=True), codebert, device, epochs=1)
        cb_feats, heuristics, _ = global_pre_extract_codebert_heuristics(train_val_loader, codebert, device)
        
    # Prepare CSV for Variance logging
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/kfold_variance_logs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fold", "Train_F1", "Train_Precision", "Val_F1", "Val_Precision"])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    
    print("\n" + "="*50)
    print(f"🚀 PHASE 2: {n_splits}-FOLD CROSS VALIDATION & OPTUNA ({mode.upper()} MODE)")
    print("="*50)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(train_val_labels)), train_val_labels)):
        print(f"\n--- FOLD {fold+1}/{n_splits} ---")
        
        # PERFECT ISOLATION: Split Train Fold into Sub_Train (80%) and Opt_Val (20%) BEFORE CNN training
        sub_train_idx, opt_val_idx = train_test_split(train_idx, test_size=0.2, random_state=42, stratify=train_val_labels[train_idx])
        
        sub_train_subset = Subset(train_val_dataset, sub_train_idx)
        opt_val_subset = Subset(train_val_dataset, opt_val_idx)
        test_subset = Subset(train_val_dataset, test_idx)
        
        sub_train_loader_shuffled = DataLoader(sub_train_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=4 if os.name == 'nt' else 8, pin_memory=True, persistent_workers=True)
        
        # Train CNN strictly on Sub_Train
        print("[*] Training CNN specifically on Sub_Train (Isolating Optuna Validation)...")
        cnn = train_cnn(sub_train_loader_shuffled, device, epochs=5)
        
        if mode == "slow":
            print("[*] Training CodeBERT LoRA specifically on Sub_Train...")
            codebert_fold = CodeBERTEmbedding(use_lora=True).to(device)
            train_codebert_lora(sub_train_loader_shuffled, codebert_fold, device, epochs=1)
            
            cnn_sub_train, cb_sub_train, h_sub_train, y_sub_train = extract_all_features(DataLoader(sub_train_subset, batch_size=batch_size, collate_fn=custom_collate, num_workers=4 if os.name == 'nt' else 8, pin_memory=True, persistent_workers=True), cnn, codebert_fold, device, "Extract Sub-Train")
            cnn_opt_val, cb_opt_val, h_opt_val, y_opt_val = extract_all_features(DataLoader(opt_val_subset, batch_size=batch_size, collate_fn=custom_collate, num_workers=4 if os.name == 'nt' else 8, pin_memory=True, persistent_workers=True), cnn, codebert_fold, device, "Extract Optuna-Val")
            cnn_test, cb_test, h_test, y_test = extract_all_features(DataLoader(test_subset, batch_size=batch_size, collate_fn=custom_collate, num_workers=4 if os.name == 'nt' else 8, pin_memory=True, persistent_workers=True), cnn, codebert_fold, device, "Extract Test")
            
            X_sub_train = np.concatenate([cnn_sub_train, cb_sub_train, h_sub_train], axis=1)
            X_opt_val = np.concatenate([cnn_opt_val, cb_opt_val, h_opt_val], axis=1)
            X_test = np.concatenate([cnn_test, cb_test, h_test], axis=1)
            
        else:
            # Fast Mode: Extract only CNN, grab CodeBERT from global cache
            cnn_sub_train = extract_cnn_features(DataLoader(sub_train_subset, batch_size=batch_size, collate_fn=custom_collate, num_workers=4 if os.name == 'nt' else 8, pin_memory=True, persistent_workers=True), cnn, device, "Extract CNN Sub-Train")
            cnn_opt_val = extract_cnn_features(DataLoader(opt_val_subset, batch_size=batch_size, collate_fn=custom_collate, num_workers=4 if os.name == 'nt' else 8, pin_memory=True, persistent_workers=True), cnn, device, "Extract CNN Optuna-Val")
            cnn_test = extract_cnn_features(DataLoader(test_subset, batch_size=batch_size, collate_fn=custom_collate, num_workers=4 if os.name == 'nt' else 8, pin_memory=True, persistent_workers=True), cnn, device, "Extract CNN Test")
            
            # Heuristics array could be 1D or 2D, safely reshape
            h_sub = heuristics[sub_train_idx]
            h_opt = heuristics[opt_val_idx]
            h_tst = heuristics[test_idx]
            if len(h_sub.shape) == 1:
                h_sub = h_sub.reshape(-1, 1)
                h_opt = h_opt.reshape(-1, 1)
                h_tst = h_tst.reshape(-1, 1)
                
            X_sub_train = np.concatenate([cnn_sub_train, cb_feats[sub_train_idx], h_sub], axis=1)
            y_sub_train = train_val_labels[sub_train_idx]
            
            X_opt_val = np.concatenate([cnn_opt_val, cb_feats[opt_val_idx], h_opt], axis=1)
            y_opt_val = train_val_labels[opt_val_idx]
            
            X_test = np.concatenate([cnn_test, cb_feats[test_idx], h_tst], axis=1)
            y_test = train_val_labels[test_idx]
            
        # Re-combine Sub_Train and Opt_Val into Full Fold Train for final evaluation
        X_train_full = np.concatenate([X_sub_train, X_opt_val], axis=0)
        y_train_full = np.concatenate([y_sub_train, y_opt_val], axis=0)
        
        # Apply Distance Normalisation (Standard Scaling) prior to SMOTE
        scaler = StandardScaler()
        X_sub_train_scaled = scaler.fit_transform(X_sub_train)
        X_opt_val_scaled = scaler.transform(X_opt_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply SMOTE strictly to Sub_Train for Optuna
        print(f"[*] Raw Sub-Train Shape: {X_sub_train_scaled.shape} | Benign: {np.sum(y_sub_train==0)}, Phishing: {np.sum(y_sub_train==1)}")
        smote = SMOTE(random_state=42)
        X_sub_train_smote, y_sub_train_smote = smote.fit_resample(X_sub_train_scaled, y_sub_train)
        
        print("[*] Running Optuna Parameter Optimization...")
        optuna.logging.set_verbosity(optuna.logging.WARNING) # Suppress massive output
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_sub_train_smote, y_sub_train_smote, X_opt_val_scaled, y_opt_val), n_trials=n_optuna_trials)
        
        best = study.best_trial.params
        print(f"[*] Optuna Best Trial F1: {study.best_trial.value:.4f}")
        
        best_xgb_params = {
            'max_depth': best['xgb_max_depth'],
            'learning_rate': best['xgb_lr'],
            'n_estimators': best['xgb_n_estimators'],
            'subsample': best['xgb_subsample'],
            'tree_method': 'hist',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Re-scale Full Fold Train and test for final evaluation
        X_train_full_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled_final = scaler.transform(X_test)

        # Final Train on Full SMOTE Fold
        X_train_smote, y_train_smote = smote.fit_resample(X_train_full_scaled, y_train_full)
        meta = MetaClassifier(use_logistic_regression=True, xgb_params=best_xgb_params)
        meta.train(X_train_smote, y_train_smote)
        
        if fold == n_splits - 1:
            os.makedirs("weights", exist_ok=True)
            meta.save("weights/meta_classifier.pkl")
            torch.save(cnn.state_dict(), "weights/cnn_trained.pt")
            if mode == "slow":
                torch.save(codebert_fold.state_dict(), "weights/codebert_trained.pt")
            else:
                torch.save(codebert.state_dict(), "weights/codebert_trained.pt")
                
        # Evaluate on Validation (Test Fold)
        preds_val = [meta.predict(x) for x in X_test_scaled_final]
        acc_val = accuracy_score(y_test, preds_val)
        prec_val = precision_score(y_test, preds_val, zero_division=0)
        rec_val = recall_score(y_test, preds_val, zero_division=0)
        f1_val = f1_score(y_test, preds_val, zero_division=0)
        
        auc_val = roc_auc_score(y_test, preds_val)
        mcc_val = matthews_corrcoef(y_test, preds_val)
        cm_val = confusion_matrix(y_test, preds_val)
        tn, fp, fn, tp = cm_val.ravel()
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Evaluate on Train Fold (Method 2 Logging)
        preds_train = [meta.predict(x) for x in X_train_full_scaled] # Using full train (un-smoted) for realistic training precision
        prec_train = precision_score(y_train_full, preds_train, zero_division=0)
        f1_train = f1_score(y_train_full, preds_train, zero_division=0)
        
        with open("metrics/kfold_variance_logs.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([fold+1, f"{f1_train:.4f}", f"{prec_train:.4f}", f"{f1_val:.4f}", f"{prec_val:.4f}"])
        
        print(f"[+] Fold {fold+1} Metrics: F1: {f1_val*100:.2f}% | Prec: {prec_val*100:.2f}% | Rec: {rec_val*100:.2f}% | Acc: {acc_val*100:.2f}% | AUC: {auc_val:.4f} | MCC: {mcc_val:.4f} | FPR: {fpr_val*100:.2f}%")
        fold_metrics.append((f1_val, prec_val, rec_val, acc_val, auc_val, mcc_val, fpr_val))
        
    print("\n" + "="*50)
    print("🚀 FINAL 5-FOLD CROSS VALIDATION RESULTS")
    print("="*50)
    f1s, precs, recs, accs, aucs, mccs, fprs = zip(*fold_metrics)
    
    print(f"Average Accuracy:  {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"Average Precision: {np.mean(precs)*100:.2f}% ± {np.std(precs)*100:.2f}%")
    print(f"Average Recall:    {np.mean(recs)*100:.2f}% ± {np.std(recs)*100:.2f}%")
    print(f"Average F1-Score:  {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    print(f"Average ROC-AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Average MCC:       {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")
    print(f"Average FPR:       {np.mean(fprs)*100:.2f}% ± {np.std(fprs)*100:.2f}%")

if __name__ == "__main__":
    start_time = time.time()
    train_model(batch_size=32, n_optuna_trials=10, n_splits=5)
    end_time = time.time()
    
    total_time = end_time - start_time
    mins, secs = divmod(total_time, 60)
    print("\n" + "="*50)
    print(f"⏱️ TOTAL PIPELINE EXECUTION TIME: {int(mins)} minutes and {int(secs)} seconds")
    print("\n[!] Training complete!")
    print("[!] Run 'python Check_for_overfitting.py' to evaluate the zero-day vault and check for overfitting.")
    print("="*50)