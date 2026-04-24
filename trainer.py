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

def train_cnn(train_loader, device, epochs=5):
    print("\n" + "="*50)
    print("🚀 PHASE 1: Training Structural CNN")
    print("="*50)
    cnn = CNN1DEmbedding(embedding_dim=64, num_filters=128).to(device)
    temp_classifier = nn.Linear(128, 1).to(device)
    optimizer = optim.Adam(list(cnn.parameters()) + list(temp_classifier.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    cnn.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_dicts, labels in tqdm(train_loader, desc=f"CNN Epoch {epoch+1}/{epochs}"):
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
            
    os.makedirs("weights", exist_ok=True)
    torch.save(cnn.state_dict(), "weights/cnn_trained.pt")
    return cnn

def pre_extract_features(dataloader, cnn, codebert, device, desc="Extracting Features"):
    cnn.eval()
    codebert.eval()
    
    cnn_feats, cb_feats, heuristics, labels_list = [], [], [], []
    
    with torch.no_grad():
        for batch_dicts, labels in tqdm(dataloader, desc=desc):
            for item, label in zip(batch_dicts, labels):
                # CNN
                cnn_input = item['cnn_input'].unsqueeze(0).to(device)
                c_emb = cnn(cnn_input).cpu().numpy().flatten()
                cnn_feats.append(c_emb)
                
                # CodeBERT
                cb_text = item['codebert_text']
                cb_emb = codebert.compute_embedding(cb_text).cpu().numpy().flatten()
                cb_feats.append(cb_emb)
                
                # Heuristics
                heuristics.append(item['heuristic'].cpu().numpy())
                labels_list.append(label.item())
                
    X = np.concatenate([
        np.array(cnn_feats), 
        np.array(cb_feats), 
        np.array(heuristics).reshape(-1, 1)
    ], axis=1)
    return X, np.array(labels_list)

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

def train_model(batch_size=4, n_optuna_trials=30):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    dirs = {'phishing': 'dataset/raw_html/phishing', 'benign': 'dataset/raw_html/benign'}
    dataset = PhishingDataset(dirs, undersample_benign=False)
    if len(dataset) == 0:
        print("Dataset is empty.")
        return
        
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    extract_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    extract_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    
    # 1. Train CNN
    cnn = train_cnn(train_loader, device, epochs=5)
    codebert = CodeBERTEmbedding().to(device)
    
    # 2. Extract Features
    print("\n" + "="*50)
    print("🚀 PHASE 2: Pre-Extracting Deep Learning Features")
    print("="*50)
    X_train, y_train = pre_extract_features(extract_train_loader, cnn, codebert, device, "Extracting Train")
    X_val, y_val = pre_extract_features(extract_val_loader, cnn, codebert, device, "Extracting Val")
    X_test, y_test = pre_extract_features(test_loader, cnn, codebert, device, "Extracting Test")
    
    # 3. Optuna Optimization for XGBoost
    print("\n" + "="*50)
    print("🚀 PHASE 3: Optuna XGBoost Meta-Classifier Optimization")
    print("="*50)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_optuna_trials)
    
    print("\n[+] Best Optuna Trial:")
    print(f"  F1-Score: {study.best_value:.4f}")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
        
    # 4. Final Meta-Classifier
    print("\n🚀 FINAL TRAINING WITH BEST PARAMS...")
    meta = MetaClassifier(use_logistic_regression=True, xgb_params=study.best_trial.params)
    meta.train(X_train, y_train)
    meta.save("weights/meta_classifier.pkl")
    
    print("\n" + "="*50)
    print("🚀 FINAL METRICS ON UNSEEN TEST SET (15%)")
    print("="*50)
    all_preds = [meta.predict(x) for x in X_test]
    
    print(f"Accuracy:  {accuracy_score(y_test, all_preds)*100:.2f}%")
    print(f"Precision: {precision_score(y_test, all_preds, zero_division=0)*100:.2f}%")
    print(f"Recall:    {recall_score(y_test, all_preds, zero_division=0)*100:.2f}%")
    print(f"F1-Score:  {f1_score(y_test, all_preds, zero_division=0)*100:.2f}%")
    cm = confusion_matrix(y_test, all_preds)
    print("\nConfusion Matrix:")
    print(f"True Negatives:  {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives:  {cm[1][1]}")

if __name__ == "__main__":
    train_model(batch_size=4, n_optuna_trials=30)
