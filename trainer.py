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
import optuna

def set_seed(seed=42):
    """Locks all random seeds for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

def train_codebert_lora(train_loader, device, epochs=3):
    print("\n" + "="*50)
    print("🚀 PHASE 1.5: Fine-Tuning CodeBERT with PEFT (LoRA)")
    print("="*50)
    
    codebert = CodeBERTEmbedding(use_lora=True).to(device)
    codebert.train_lora_mode()
    
    temp_classifier = nn.Linear(768, 1).to(device)
    optimizer = optim.AdamW(list(codebert.parameters()) + list(temp_classifier.parameters()), lr=2e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"LoRA Epoch {epoch+1}/{epochs}", unit="batch")
        for batch_dicts, labels in progress_bar:
            labels = labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            
            batch_loss = 0.0
            
            # Gradient Accumulation: Forward/Backward per item to save massive amounts of VRAM
            for i, item in enumerate(batch_dicts):
                cb_text = item['codebert_text']
                emb = codebert.compute_embedding(cb_text).unsqueeze(0).to(device)
                
                logit = temp_classifier(emb)
                single_label = labels[i:i+1]
                
                loss = criterion(logit, single_label)
                loss = loss / len(batch_dicts) # Scale loss since we are accumulating
                loss.backward() # This frees the activation graph immediately!
                
                batch_loss += loss.item() * len(batch_dicts)
                
            optimizer.step()
            
            # Clear backend cache aggressively to prevent MPS OOM
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            epoch_loss += batch_loss
            progress_bar.set_postfix(loss=f"{batch_loss / len(batch_dicts):.4f}")
            
    print("[+] LoRA Fine-tuning complete. Saving adapter weights...")
    os.makedirs("weights/lora_adapter", exist_ok=True)
    codebert.codebert.save_pretrained("weights/lora_adapter")
    return codebert

def pre_extract_codebert_features(dataloader, codebert, device):
    codebert.eval_lora_and_freeze()
    cb_features, heuristics, labels_list = [], [], []
    
    with torch.no_grad():
        for batch_dicts, labels in tqdm(dataloader, desc="Extracting Static CodeBERT Features"):
            for item, label in zip(batch_dicts, labels):
                cb_text = item['codebert_text']
                emb = codebert.compute_embedding(cb_text)
                cb_features.append(emb.cpu().numpy().flatten())
                heuristics.append(item['heuristic'].cpu().numpy())
                labels_list.append(label.item())
                
    return np.array(cb_features), np.array(heuristics), np.array(labels_list)

def objective(trial, train_loader, extract_train_loader, extract_val_loader, pre_train, pre_val, device):
    # 1. Suggest CNN Hyperparameters
    lr = trial.suggest_float("cnn_lr", 1e-4, 5e-3, log=True)
    num_filters = trial.suggest_categorical("cnn_filters", [64, 128])
    embedding_dim = trial.suggest_categorical("cnn_emb_dim", [32, 64])
    
    cnn = CNN1DEmbedding(embedding_dim=embedding_dim, num_filters=num_filters).to(device)
    temp_classifier = nn.Linear(128, 1).to(device)
    optimizer = optim.Adam(list(cnn.parameters()) + list(temp_classifier.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Train CNN for limited epochs during Optuna trials
    cnn.train()
    for epoch in range(2):
        for batch_dicts, labels in train_loader:
            labels = labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            batch_logits = []
            for item in batch_dicts:
                cnn_input = item['cnn_input'].unsqueeze(0).to(device)
                emb = cnn(cnn_input)
                batch_logits.append(emb)
            batch_tensor = torch.cat(batch_logits, dim=0)
            loss = criterion(temp_classifier(batch_tensor), labels)
            loss.backward()
            optimizer.step()
            
    # Extract CNN features using non-shuffled loaders to align with precomputed arrays
    cnn.eval()
    def get_cnn_features(loader):
        feats = []
        with torch.no_grad():
            for batch_dicts, _ in loader:
                for item in batch_dicts:
                    cnn_input = item['cnn_input'].unsqueeze(0).to(device)
                    emb = cnn(cnn_input)
                    feats.append(emb.cpu().numpy().flatten())
        return np.array(feats)
        
    cnn_train = get_cnn_features(extract_train_loader)
    cnn_val = get_cnn_features(extract_val_loader)
    
    # Concat features: CNN (128) + CodeBERT (768) + Heuristics (1)
    X_train = np.concatenate([cnn_train, pre_train[0], pre_train[1].reshape(-1, 1)], axis=1)
    X_val = np.concatenate([cnn_val, pre_val[0], pre_val[1].reshape(-1, 1)], axis=1)
    
    # 2. Suggest XGBoost Hyperparameters
    xgb_params = {
        'max_depth': trial.suggest_int('xgb_max_depth', 2, 5),
        'learning_rate': trial.suggest_float('xgb_lr', 1e-3, 0.3, log=True),
        'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 150),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0)
    }
    
    meta = MetaClassifier(use_logistic_regression=False, xgb_params=xgb_params)
    meta.train(X_train, pre_train[2])
    
    preds = []
    for x in X_val:
        preds.append(meta.predict(x))
        
    f1 = f1_score(pre_val[2], preds, zero_division=0)
    return f1

def train_model(batch_size=4, n_optuna_trials=10):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    dirs = {'phishing': 'dataset/raw_html/phishing', 'benign': 'dataset/raw_html/benign'}
    dataset = PhishingDataset(dirs, undersample_benign=False)
    dataset_len = len(dataset)
    if dataset_len == 0:
        print("Dataset is empty.")
        return
        
    train_size = int(0.7 * dataset_len)
    val_size = int(0.15 * dataset_len)
    test_size = dataset_len - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Shuffled loader for CNN training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    # Non-shuffled loaders for feature extraction alignment
    extract_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    extract_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    
    # --- PHASE 1.5: LoRA ---
    # In a real run, you only need to run this once.
    if not os.path.exists("weights/lora_adapter"):
        codebert = train_codebert_lora(train_loader, device, epochs=2)
    else:
        print("[+] Loading existing LoRA adapter...")
        codebert = CodeBERTEmbedding(use_lora=True).to(device)
        from peft import PeftModel
        codebert.codebert = PeftModel.from_pretrained(codebert.codebert.get_base_model(), "weights/lora_adapter")
        codebert.to(device)

    # Pre-extract CodeBERT embeddings since they are frozen during Optuna trials
    print("\n" + "="*50)
    print("🚀 PRE-EXTRACTING STATIC CODEBERT FEATURES FOR OPTUNA")
    print("="*50)
    pre_train = pre_extract_codebert_features(extract_train_loader, codebert, device)
    pre_val = pre_extract_codebert_features(extract_val_loader, codebert, device)
    pre_test = pre_extract_codebert_features(test_loader, codebert, device)
    
    # --- OPTUNA STUDY ---
    print("\n" + "="*50)
    print("🚀 LAUNCHING OPTUNA: CNN + XGBoost Joint Optimization")
    print("="*50)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_loader, extract_train_loader, extract_val_loader, pre_train, pre_val, device), n_trials=n_optuna_trials)
    
    print("\n[+] Best Optuna Trial:")
    print(f"  F1-Score: {study.best_value:.4f}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
        
    print("\n🚀 FINAL TRAINING WITH BEST PARAMS...")
    best = study.best_trial.params
    
    # Retrain CNN fully
    cnn = CNN1DEmbedding(embedding_dim=best['cnn_emb_dim'], num_filters=best['cnn_filters']).to(device)
    temp_classifier = nn.Linear(128, 1).to(device)
    optimizer = optim.Adam(list(cnn.parameters()) + list(temp_classifier.parameters()), lr=best['cnn_lr'])
    criterion = nn.BCEWithLogitsLoss()
    
    cnn.train()
    for epoch in range(5):
        for batch_dicts, labels in tqdm(train_loader, desc=f"Final CNN Epoch {epoch+1}/5"):
            labels = labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            batch_logits = []
            for item in batch_dicts:
                cnn_input = item['cnn_input'].unsqueeze(0).to(device)
                batch_logits.append(cnn(cnn_input))
            loss = criterion(temp_classifier(torch.cat(batch_logits, dim=0)), labels)
            loss.backward()
            optimizer.step()
            
    os.makedirs("weights", exist_ok=True)
    torch.save(cnn.state_dict(), "weights/cnn_trained.pt")
    
    # Final Extract
    cnn.eval()
    def get_final_cnn(loader):
        feats = []
        with torch.no_grad():
            for batch_dicts, _ in loader:
                for item in batch_dicts:
                    cnn_input = item['cnn_input'].unsqueeze(0).to(device)
                    feats.append(cnn(cnn_input).cpu().numpy().flatten())
        return np.array(feats)
        
    cnn_train_final = get_final_cnn(extract_train_loader)
    cnn_test_final = get_final_cnn(test_loader)
    
    X_train_final = np.concatenate([cnn_train_final, pre_train[0], pre_train[1].reshape(-1, 1)], axis=1)
    X_test_final = np.concatenate([cnn_test_final, pre_test[0], pre_test[1].reshape(-1, 1)], axis=1)
    
    xgb_params = {
        'max_depth': best['xgb_max_depth'], 'learning_rate': best['xgb_lr'],
        'n_estimators': best['xgb_n_estimators'], 'subsample': best['xgb_subsample']
    }
    meta = MetaClassifier(use_logistic_regression=True, xgb_params=xgb_params)
    meta.train(X_train_final, pre_train[2])
    meta.save("weights/meta_classifier.pkl")
    
    # Test Metrics
    print("\n" + "="*50)
    print("🚀 FINAL METRICS ON UNSEEN TEST SET (15%)")
    print("="*50)
    all_preds = [meta.predict(x) for x in X_test_final]
    
    acc = accuracy_score(pre_test[2], all_preds)
    prec = precision_score(pre_test[2], all_preds, zero_division=0)
    rec = recall_score(pre_test[2], all_preds, zero_division=0)
    f1 = f1_score(pre_test[2], all_preds, zero_division=0)
    cm = confusion_matrix(pre_test[2], all_preds)
    
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"True Negatives:  {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives:  {cm[1][1]}")

if __name__ == "__main__":
    train_model(batch_size=4, n_optuna_trials=10)
