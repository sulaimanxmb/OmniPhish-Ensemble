import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset_loader import PhishingDataset, custom_collate
from cnn_model import CNN1DEmbedding
from transformer_model import CodeBERTEmbedding
from classifier import MetaClassifier
import os
import random
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE

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

set_seed(42)

def global_pre_extract_codebert_heuristics(dataloader, codebert, device):
    """Extracts CodeBERT and Heuristic features for the ENTIRE dataset ONCE to save time across folds."""
    codebert.eval()
    cb_feats, heuristics, labels_list = [], [], []
    
    with torch.no_grad():
        for batch_dicts, labels in tqdm(dataloader, desc="Global Extraction: CodeBERT & Heuristics"):
            for item, label in zip(batch_dicts, labels):
                cb_emb = codebert.compute_embedding(item['codebert_text']).cpu().numpy().flatten()
                cb_feats.append(cb_emb)
                heuristics.append(item['heuristic'].cpu().numpy())
                labels_list.append(label.item())
                
    return np.array(cb_feats), np.array(heuristics), np.array(labels_list)

def train_cnn(train_loader, device, epochs=5):
    cnn = CNN1DEmbedding(embedding_dim=64, num_filters=128).to(device)
    temp_classifier = nn.Linear(128, 1).to(device)
    optimizer = optim.Adam(list(cnn.parameters()) + list(temp_classifier.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    cnn.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_dicts, labels in tqdm(train_loader, desc=f"CNN Epoch {epoch+1}/{epochs}", leave=False):
            labels = labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            batch_logits = []
            for item in batch_dicts:
                cnn_input = item['cnn_input'].unsqueeze(0).to(device)
                batch_logits.append(cnn(cnn_input))
            loss = criterion(temp_classifier(torch.cat(batch_logits, dim=0)), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    return cnn

def extract_cnn_features(dataloader, cnn, device, desc):
    cnn.eval()
    cnn_feats = []
    with torch.no_grad():
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
        'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0)
    }
    meta = MetaClassifier(use_logistic_regression=False, xgb_params=xgb_params)
    meta.train(X_train, y_train)
    preds = [meta.predict(x) for x in X_val]
    return f1_score(y_val, preds, zero_division=0)

def train_model(batch_size=4, n_optuna_trials=10, n_splits=5):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    dirs = {'phishing': 'dataset/raw_html/phishing', 'benign': 'dataset/raw_html/benign'}
    dataset = PhishingDataset(dirs, undersample_benign=False)
    if len(dataset) == 0:
        print("Dataset is empty.")
        return
        
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    codebert = CodeBERTEmbedding().to(device)
    
    print("\n" + "="*50)
    print("🚀 PHASE 1: GLOBAL PRE-EXTRACTION")
    print("="*50)
    # Extract CodeBERT and Heuristics once to prevent massive overhead inside folds
    cb_feats, heuristics, all_labels = global_pre_extract_codebert_heuristics(full_loader, codebert, device)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    
    print("\n" + "="*50)
    print(f"🚀 PHASE 2: {n_splits}-FOLD CROSS VALIDATION & SMOTE")
    print("="*50)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        print(f"\n--- FOLD {fold+1}/{n_splits} ---")
        
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        train_loader_shuffled = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        train_loader_seq = DataLoader(train_subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        test_loader_seq = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        
        # Train CNN on this specific fold to prevent data leakage
        print("[*] Training Fold-Specific CNN...")
        cnn = train_cnn(train_loader_shuffled, device, epochs=5)
        
        # Extract CNN Features
        cnn_train = extract_cnn_features(train_loader_seq, cnn, device, "Extract CNN Train")
        cnn_test = extract_cnn_features(test_loader_seq, cnn, device, "Extract CNN Test")
        
        # Assemble Final Hybrid Vectors
        X_train = np.concatenate([cnn_train, cb_feats[train_idx], heuristics[train_idx].reshape(-1, 1)], axis=1)
        y_train = all_labels[train_idx]
        
        X_test = np.concatenate([cnn_test, cb_feats[test_idx], heuristics[test_idx].reshape(-1, 1)], axis=1)
        y_test = all_labels[test_idx]
        
        # Apply SMOTE to training fold exclusively
        print(f"[*] Raw Train Shape: {X_train.shape} | Benign: {np.sum(y_train==0)}, Phishing: {np.sum(y_train==1)}")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print(f"[*] SMOTE Applied   : {X_train_smote.shape} | Benign: {np.sum(y_train_smote==0)}, Phishing: {np.sum(y_train_smote==1)}")
        
        # Optuna Optimization
        X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
            X_train_smote, y_train_smote, test_size=0.2, random_state=42, stratify=y_train_smote
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_opt_train, y_opt_train, X_opt_val, y_opt_val), n_trials=n_optuna_trials)
        
        # Final Train and Test for this fold
        meta = MetaClassifier(use_logistic_regression=True, xgb_params=study.best_trial.params)
        meta.train(X_train_smote, y_train_smote)
        
        # Only save weights of the final fold (or could omit)
        if fold == n_splits - 1:
            os.makedirs("weights", exist_ok=True)
            meta.save("weights/meta_classifier.pkl")
            torch.save(cnn.state_dict(), "weights/cnn_trained.pt")
            
        preds = [meta.predict(x) for x in X_test]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        print(f"[+] Fold {fold+1} Metrics: F1: {f1*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | Acc: {acc*100:.2f}%")
        fold_metrics.append((f1, prec, rec, acc))
        
    print("\n" + "="*50)
    print("🚀 FINAL 5-FOLD CROSS VALIDATION RESULTS")
    print("="*50)
    f1s, precs, recs, accs = zip(*fold_metrics)
    
    print(f"Average Accuracy:  {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"Average Precision: {np.mean(precs)*100:.2f}% ± {np.std(precs)*100:.2f}%")
    print(f"Average Recall:    {np.mean(recs)*100:.2f}% ± {np.std(recs)*100:.2f}%")
    print(f"Average F1-Score:  {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")

if __name__ == "__main__":
    train_model(batch_size=4, n_optuna_trials=10, n_splits=5)
