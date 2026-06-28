import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import os
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_visualization(filename, model_type=None):
    if model_type and filename != 'baseline_comparison.png':
        filename = f"{model_type}_{filename}"
        
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"    [!] Deleted old duplicate: '{filename}'")
        except Exception:
            pass
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"  [+] Saved '{filepath}'")

from omniphish.dataset_loader import PhishingDataset, custom_collate
from omniphish.cnn_model import CNN1DEmbedding
from omniphish.gnn_model import GNNEmbedding
from omniphish.transformer_model import CodeBERTEmbedding
from omniphish.classifier import MetaClassifier

def get_feature_names(model_type):
    struct_prefix = "CNN" if model_type == "cnn" else "GNN"
    return [f"{struct_prefix}_Feat_{i}" for i in range(128)] + \
           [f"CodeBERT_Feat_{i}" for i in range(768)] + \
           ["Heur_DOM_Depth", "Heur_Suspicious_Action", "Heur_URL_Score"]

def generate_xgboost_importance(model_type):
    print(f"\n[*] Generating XGBoost Feature Importance Plot ({model_type.upper()})...")
    meta = MetaClassifier()
    weights_dir = f"weights_{model_type}" if os.path.exists(f"weights_{model_type}") else "weights"
    try:
        meta.load(f"{weights_dir}/meta_classifier.pkl")
    except Exception as e:
        print(f"[!] Failed to load Meta-Classifier: {e}")
        return

    meta.xgb_model.get_booster().feature_names = get_feature_names(model_type)

    plt.figure(figsize=(10, 8))
    xgb.plot_importance(meta.xgb_model, max_num_features=20, importance_type="weight", 
                        title=f"Top 20 Ensemble Features (XGBoost - {model_type.upper()})", 
                        color="#1f77b4", grid=False)
    plt.tight_layout()
    save_visualization('xgboost_importance.png', model_type)

def generate_smote_pca(model_type):
    print(f"\n[*] Initializing SMOTE Synthetic Data Generator ({model_type.upper()})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    dirs = {'phishing': 'dataset/raw_html/phishing', 'benign': 'dataset/raw_html/benign'}
    dataset = PhishingDataset(dirs, undersample_benign=False)
    
    benign_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    phish_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
    
    selected_benign = random.sample(benign_indices, min(50, len(benign_indices)))
    selected_phish = random.sample(phish_indices, min(250, len(phish_indices)))
    
    imbalanced_idx = selected_benign + selected_phish
    imbalanced_subset = Subset(dataset, imbalanced_idx)
    loader = DataLoader(imbalanced_subset, batch_size=32, shuffle=False, collate_fn=custom_collate, num_workers=4 if os.name == 'nt' else 8, pin_memory=True, persistent_workers=True)
    
    weights_dir = f"weights_{model_type}" if os.path.exists(f"weights_{model_type}") else "weights"
    
    struct_model = None
    if model_type == "cnn":
        struct_model = CNN1DEmbedding(embedding_dim=64, num_filters=128).to(device)
        try:
            struct_model.load_state_dict(torch.load(f"{weights_dir}/cnn_trained.pt", map_location=device))
        except Exception as e:
            print(f"[!] Failed to load CNN weights. Error: {e}")
            return
    elif model_type == "gnn":
        struct_model = GNNEmbedding(embedding_dim=64, hidden_dim=64, dropout=0.5).to(device)
        try:
            struct_model.load_state_dict(torch.load(f"{weights_dir}/gnn_trained.pt", map_location=device))
        except Exception as e:
            print(f"[!] Failed to load GNN weights. Error: {e}")
            return
    struct_model.eval()
    
    codebert = CodeBERTEmbedding(use_lora=True).to(device)
    codebert.load_state_dict(torch.load(f"{weights_dir}/codebert_trained.pt", map_location=device))
    codebert.eval()
    
    print(f"[*] Extracting Mathematical Embeddings for highly imbalanced samples ({model_type.upper()})...")
    struct_feats, cb_feats, heuristics, labels_list = [], [], [], []
    
    with torch.no_grad():
        for batch_dicts, labels in tqdm(loader, desc="SMOTE Extraction"):
            for item, label in zip(batch_dicts, labels):
                if model_type == "cnn":
                    s_emb = struct_model(item['cnn_input'].unsqueeze(0).to(device)).cpu().numpy().flatten()
                elif model_type == "gnn":
                    gnn_nodes = item['gnn_nodes'].unsqueeze(0).to(device)
                    gnn_adj = item['gnn_adj'].unsqueeze(0).to(device)
                    s_emb = struct_model(gnn_nodes, gnn_adj).cpu().numpy().flatten()
                    
                cb_emb = codebert.compute_embedding(item['codebert_text']).cpu().numpy().flatten()
                
                struct_feats.append(s_emb)
                cb_feats.append(cb_emb)
                heuristics.append(item['heuristic'].cpu().numpy())
                labels_list.append(label.item())
                
    struct_feats = np.array(struct_feats)
    cb_feats = np.array(cb_feats)
    heuristics = np.array(heuristics)
    y_true = np.array(labels_list, dtype=int)
    
    if len(heuristics.shape) == 1:
        heuristics = heuristics.reshape(-1, 1)
        
    X_imbalanced = np.concatenate([struct_feats, cb_feats, heuristics], axis=1)
    
    # Apply distance normalisation
    scaler = StandardScaler()
    X_imbalanced_scaled = scaler.fit_transform(X_imbalanced)
    
    print("[*] Injecting Synthetic Phishing Samples via SMOTE...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_imbalanced_scaled, y_true)
    
    print("[*] Compressing 899 Dimensions -> 2 Dimensions via PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_balanced)
    
    X_imb_pca = pca.transform(X_imbalanced)
    X_bal_pca = pca.transform(X_balanced)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    axes[0].scatter(X_imb_pca[y_true == 0, 0], X_imb_pca[y_true == 0, 1], c='blue', alpha=0.6, label='Benign (0)', edgecolors='w', s=50)
    axes[0].scatter(X_imb_pca[y_true == 1, 0], X_imb_pca[y_true == 1, 1], c='red', alpha=0.9, label='Phishing (1)', edgecolors='w', s=50)
    axes[0].set_title('Before SMOTE (Zero-Day Imbalance)', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(X_bal_pca[y_balanced == 0, 0], X_bal_pca[y_balanced == 0, 1], c='blue', alpha=0.6, label='Benign (0)', edgecolors='w', s=50)
    axes[1].scatter(X_bal_pca[y_balanced == 1, 0], X_bal_pca[y_balanced == 1, 1], c='red', alpha=0.9, label='Phishing (Synthetic & Real)', edgecolors='w', s=50)
    axes[1].set_title('After SMOTE (Synthetically Balanced)', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'SMOTE Effectiveness in OmniPhish 899-D Spatial Memory ({model_type.upper()})', fontweight='bold', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_visualization('smote_pca_comparison.png', model_type)


def generate_inference_based_graphs(choices, model_type):
    print(f"\n[*] Initializing Zero-Day Vault Inference ({model_type.upper()})...")
    weights_dir = f"weights_{model_type}" if os.path.exists(f"weights_{model_type}") else "weights"
    
    if not os.path.exists(f"{weights_dir}/vault_indices.npy"):
        if os.path.exists("weights/vault_indices.npy"):
            weights_dir = "weights"
        else:
            print(f"[!] Error: 'vault_indices.npy' not found in {weights_dir}/.")
            return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    vault_idx = np.load(f"{weights_dir}/vault_indices.npy")
    print(f"[*] Loaded {len(vault_idx)} isolated Zero-Day samples.")
    
    dirs = {'phishing': 'dataset/raw_html/phishing', 'benign': 'dataset/raw_html/benign'}
    dataset = PhishingDataset(dirs, undersample_benign=False)
    vault_subset = Subset(dataset, vault_idx)
    vault_loader = DataLoader(vault_subset, batch_size=32, shuffle=False, collate_fn=custom_collate, num_workers=4 if os.name == 'nt' else 8, pin_memory=True, persistent_workers=True)
    
    print(f"[*] Loading Neural Networks for {model_type.upper()}...")
    struct_model = None
    if model_type == "cnn":
        struct_model = CNN1DEmbedding(embedding_dim=64, num_filters=128).to(device)
        try:
            struct_model.load_state_dict(torch.load(f"{weights_dir}/cnn_trained.pt", map_location=device))
        except Exception as e:
            print(f"[!] Failed to load CNN weights. Error: {e}")
            return
    elif model_type == "gnn":
        struct_model = GNNEmbedding(embedding_dim=64, hidden_dim=64, dropout=0.5).to(device)
        try:
            struct_model.load_state_dict(torch.load(f"{weights_dir}/gnn_trained.pt", map_location=device))
        except Exception as e:
            print(f"[!] Failed to load GNN weights. Error: {e}")
            return
    struct_model.eval()
    
    codebert = CodeBERTEmbedding(use_lora=True).to(device)
    codebert.load_state_dict(torch.load(f"{weights_dir}/codebert_trained.pt", map_location=device))
    codebert.eval()
    
    meta = MetaClassifier()
    meta.load(f"{weights_dir}/meta_classifier.pkl")
    
    print(f"[*] Performing Forward Pass (Extracting 899-D Mathematical Embeddings - {model_type.upper()})...")
    struct_feats, cb_feats, heuristics, labels_list = [], [], [], []
    
    with torch.no_grad():
        for batch_dicts, labels in tqdm(vault_loader, desc="Inference Extraction"):
            for item, label in zip(batch_dicts, labels):
                if model_type == "cnn":
                    s_emb = struct_model(item['cnn_input'].unsqueeze(0).to(device)).cpu().numpy().flatten()
                elif model_type == "gnn":
                    gnn_nodes = item['gnn_nodes'].unsqueeze(0).to(device)
                    gnn_adj = item['gnn_adj'].unsqueeze(0).to(device)
                    s_emb = struct_model(gnn_nodes, gnn_adj).cpu().numpy().flatten()
                    
                cb_emb = codebert.compute_embedding(item['codebert_text']).cpu().numpy().flatten()
                
                struct_feats.append(s_emb)
                cb_feats.append(cb_emb)
                heuristics.append(item['heuristic'].cpu().numpy())
                labels_list.append(label.item())
                
    struct_feats = np.array(struct_feats)
    cb_feats = np.array(cb_feats)
    heuristics = np.array(heuristics)
    y_true = np.array(labels_list, dtype=int)
    
    if len(heuristics.shape) == 1:
        heuristics = heuristics.reshape(-1, 1)
        
    X_vault = np.concatenate([struct_feats, cb_feats, heuristics], axis=1)
    
    print("[*] Applying Distance Normalization (Standard Scaling)...")
    import pickle
    try:
        with open(f"{weights_dir}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        X_vault = scaler.transform(X_vault)
    except Exception as e:
        print(f"[!] Failed to load StandardScaler: {e}")
    
    print("[*] Generating XGBoost Predictions...")
    y_pred = [meta.predict(x) for x in X_vault]
    y_probs = [meta.predict_proba(x) for x in X_vault]
    
    # 2. Confusion Matrix
    if '2' in choices:
        print("\n[*] Generating Confusion Matrix...")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Benign', 'Predicted Phishing'],
                    yticklabels=['Actual Benign', 'Actual Phishing'])
        plt.title('Zero-Day Vault Confusion Matrix', fontweight='bold', pad=15)
        plt.tight_layout()
        save_visualization('vault_confusion_matrix.png', model_type)

    # 3. PCA
    if '3' in choices:
        print("\n[*] Compressing 899 Dimensions -> 2 Dimensions via PCA...")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_vault)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap=plt.cm.coolwarm, alpha=0.6, edgecolors='w', s=50)
        plt.title(f'PCA Projection of 899-D OmniPhish Latent Space ({model_type.upper()})', fontweight='bold', pad=15)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        cbar = plt.colorbar(scatter, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Benign (0)', 'Phishing (1)'])
        plt.grid(True, alpha=0.3)
        save_visualization('pca_clusters.png', model_type)

    # 4. t-SNE
    if '4' in choices:
        print("\n[*] Compressing 899 Dimensions -> 2 Dimensions via t-SNE (This may take a minute)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_vault)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap=plt.cm.coolwarm, alpha=0.6, edgecolors='w', s=50)
        plt.title('t-SNE Spatial Clustering of Zero-Day Threats', fontweight='bold', pad=15)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        cbar = plt.colorbar(scatter, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Benign (0)', 'Phishing (1)'])
        plt.grid(True, alpha=0.3)
        save_visualization('tsne_clusters.png')

    # 5. ROC Curve
    if '5' in choices:
        print("\n[*] Generating ROC Curve & AUC...")
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC)', fontweight='bold', pad=15)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        save_visualization('vault_roc_curve.png')

    # 6. Precision-Recall Curve
    if '6' in choices:
        print("\n[*] Generating Precision-Recall Curve...")
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = average_precision_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.4f})')
        plt.xlabel('Recall', fontweight='bold')
        plt.ylabel('Precision', fontweight='bold')
        plt.title('Precision-Recall Curve', fontweight='bold', pad=15)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        save_visualization('vault_pr_curve.png')

    # 8. SHAP Explainer Values
    if '8' in choices:
        print("\n[*] Generating SHAP (SHapley Additive exPlanations) Game Theory Values...")
        print("    [!] Warning: SHAP mathematical rendering can be very computationally expensive.")
        try:
            # We sample down the vault if it's too big so SHAP doesn't freeze the Mac
            sample_size = min(200, X_vault.shape[0])
            idx = np.random.choice(X_vault.shape[0], sample_size, replace=False)
            X_shap = X_vault[idx]
            
            explainer = shap.TreeExplainer(meta.xgb_model)
            shap_values = explainer.shap_values(X_shap)
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_shap, feature_names=get_feature_names(), show=False, max_display=15)
            plt.title("SHAP Global Feature Explainability", fontweight='bold', pad=15)
            plt.tight_layout()
            save_visualization('shap_summary.png')
        except Exception as e:
            print(f"  [!] SHAP calculation failed: {e}")

    # 9. 2D Decision Boundary (PCA Surrogate)
    if '9' in choices:
        print("\n[*] Generating 2D Surrogate Decision Boundary via PCA & XGBoost...")
        print("    [!] Warning: This is an optical illusion compressing 899-D space. Not recommended for IEEE paper.")
        
        # We need PCA down to 2 components
        pca_surrogate = PCA(n_components=2, random_state=42)
        X_pca_surr = pca_surrogate.fit_transform(X_vault)
        
        # Train a lightweight surrogate XGBoost model on the 2D data
        surrogate_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        surrogate_xgb.fit(X_pca_surr, y_true)
        
        # Create a mesh grid
        x_min, x_max = X_pca_surr[:, 0].min() - 1, X_pca_surr[:, 0].max() + 1
        y_min, y_max = X_pca_surr[:, 1].min() - 1, X_pca_surr[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        
        # Predict over the grid
        Z = surrogate_xgb.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        scatter = plt.scatter(X_pca_surr[:, 0], X_pca_surr[:, 1], c=y_true, cmap=plt.cm.coolwarm, 
                              edgecolors='k', s=40, alpha=0.8)
        
        plt.title('Surrogate 2D Decision Boundary (899-D PCA Compression)', fontweight='bold', pad=15)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        cbar = plt.colorbar(scatter, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Benign (0)', 'Phishing (1)'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_visualization('surrogate_decision_boundary.png')

def generate_radar_chart():
    print("\n[*] Generating Architecture Radar Chart...")
    
    labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Hardware Efficiency']
    
    # Scale hardware efficiency so higher is better (e.g. 12GB - VRAM / 12GB * 100)
    def vram_to_score(gb):
        return max(0, ((12 - gb) / 12) * 100)
        
    models = {
        'OmniPhish (CNN)': [94.12, 96.19, 92.67, 100.0, vram_to_score(2.99)],
        'OmniPhish (GNN)': [91.74, 94.26, 94.26, 94.26, vram_to_score(3.03)],
        'HTMLPhish': [93.49, 95.42, 95.42, 95.42, vram_to_score(0.64)],
        'Longformer': [94.12, 95.99, 93.20, 98.94, vram_to_score(11.56)]
    }
    
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for color, (name, stats) in zip(colors, models.items()):
        stats = stats + stats[:1]
        ax.plot(angles, stats, color=color, linewidth=2, label=name)
        ax.fill(angles, stats, color=color, alpha=0.1)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontweight='bold', size=11)
    ax.set_ylim(70, 100)
    
    plt.title('Multi-Dimensional Architecture Comparison', size=15, fontweight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    save_visualization('architecture_radar_chart.png')

def generate_baseline_comparison():
    print("\n[*] Generating Baseline Comparison (F1-Score vs. Peak VRAM)...")
    
    # Metrics from FINAL_METRICS.md
    models = ['OmniPhish (CNN)', 'OmniPhish (GNN)', 'HTMLPhish', 'Longformer', 'Qwen2.5-Coder']
    f1_scores = [96.19, 94.26, 95.42, 95.99, 43.21]
    vrams = [2.99, 3.03, 0.64, 11.56, 9.50]  # in GB.
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar chart for F1-Score
    x = np.arange(len(models))
    width = 0.5
    
    rects1 = ax1.bar(x, f1_scores, width, label='F1-Score', color='#2ca02c')
    
    ax1.set_ylabel('F1-Score Percentage (%)', fontweight='bold', color='#2ca02c')
    ax1.tick_params(axis='y', labelcolor='#2ca02c')
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right', fontweight='bold')
    
    # Line chart for VRAM on a secondary Y-axis
    ax2 = ax1.twinx()
    line1 = ax2.plot(x, vrams, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Peak VRAM (GB)')
    
    ax2.set_ylabel('Peak VRAM (Gigabytes)', fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 13)
    
    # Add values on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
                    
    # Add values on points
    for i, txt in enumerate(vrams):
        if models[i] == 'Longformer':
            ax2.annotate(f'{txt}GB', (x[i], vrams[i]), textcoords="offset points", xytext=(0,-15), ha='center', va='top', fontweight='bold', color='red')
        else:
            ax2.annotate(f'{txt}GB', (x[i], vrams[i]), textcoords="offset points", xytext=(0,10), ha='center', va='bottom', fontweight='bold', color='red')
        
    plt.title('Baseline Architectural Comparison (F1-Score vs. Hardware Efficiency)', fontweight='bold', pad=15)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')
    
    plt.tight_layout()
    save_visualization('baseline_comparison.png')

def generate_combined_roc_pr():
    print("\n[*] Generating Combined ROC & PR Curves for OmniPhish Ensemble...")
    import os
    
    cnn_probs_file = "metrics/y_probs_cnn.npy"
    cnn_true_file = "metrics/y_true_cnn.npy"
    gnn_probs_file = "metrics/y_probs_gnn.npy"
    gnn_true_file = "metrics/y_true_gnn.npy"
    
    if not (os.path.exists(cnn_probs_file) and os.path.exists(gnn_probs_file)):
        print("[!] ERROR: Cannot find saved probabilities. Please run 'Check_for_overfitting.py' for BOTH '--model cnn' and '--model gnn' first to generate the metric arrays!")
        return
        
    y_probs_cnn = np.load(cnn_probs_file)
    y_true_cnn = np.load(cnn_true_file)
    y_probs_gnn = np.load(gnn_probs_file)
    y_true_gnn = np.load(gnn_true_file)
    
    # === COMBINED ROC CURVE ===
    plt.figure(figsize=(10, 8))
    
    fpr_cnn, tpr_cnn, _ = roc_curve(y_true_cnn, y_probs_cnn)
    auc_cnn = auc(fpr_cnn, tpr_cnn)
    plt.plot(fpr_cnn, tpr_cnn, color='#1f77b4', lw=2, label=f'OmniPhish (CNN) (AUC = {auc_cnn:.4f})')
    
    fpr_gnn, tpr_gnn, _ = roc_curve(y_true_gnn, y_probs_gnn)
    auc_gnn = auc(fpr_gnn, tpr_gnn)
    plt.plot(fpr_gnn, tpr_gnn, color='#ff7f0e', lw=2, label=f'OmniPhish (GNN) (AUC = {auc_gnn:.4f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Receiver Operating Characteristic (Zero-Day Vault)', fontweight='bold', pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_visualization('combined_roc_curve.png')
    
    # === COMBINED PR CURVE ===
    plt.figure(figsize=(10, 8))
    
    prec_cnn, rec_cnn, _ = precision_recall_curve(y_true_cnn, y_probs_cnn)
    ap_cnn = average_precision_score(y_true_cnn, y_probs_cnn)
    plt.plot(rec_cnn, prec_cnn, color='#1f77b4', lw=2, label=f'OmniPhish (CNN) (AP = {ap_cnn:.4f})')
    
    prec_gnn, rec_gnn, _ = precision_recall_curve(y_true_gnn, y_probs_gnn)
    ap_gnn = average_precision_score(y_true_gnn, y_probs_gnn)
    plt.plot(rec_gnn, prec_gnn, color='#ff7f0e', lw=2, label=f'OmniPhish (GNN) (AP = {ap_gnn:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Detection Rate)', fontweight='bold')
    plt.ylabel('Precision (Accuracy of Detection)', fontweight='bold')
    plt.title('Precision-Recall Curve (Zero-Day Vault)', fontweight='bold', pad=15)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_visualization('combined_pr_curve.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Visualizations")
    parser.add_argument("--model", type=str, choices=["cnn", "gnn"], default="cnn", help="Which model to evaluate")
    args, unknown = parser.parse_known_args()
    
    print("="*60)
    print(f"📸 OMNIPHISH VISUALIZATION SUITE ({args.model.upper()})")
    print("="*60)
    print("Select the graphs you want to generate (comma-separated):")
    print("  [0] Baseline Comparison Bar Chart (Hardcoded from FINAL_METRICS)")
    print("  [R] Architecture Radar Chart (Hardcoded from FINAL_METRICS)")
    print("  [C] Combined ROC & PR Curves (Requires evaluating BOTH models first)")
    print("  [1] XGBoost Feature Importance Plot")
    print("  [2] Confusion Matrix Heatmap")
    print("  [3] PCA Scatter Plot (899-D -> 2D)")
    print("  [4] t-SNE Scatter Plot (899-D -> 2D)")
    print("  [5] ROC Curve & AUC Score")
    print("  [6] Precision-Recall (PR) Curve")
    print("  [7] SMOTE Spatial Imbalance Demonstration (Side-by-Side PCA)")
    print("  [8] SHAP Game Theory Values (Bleeding-Edge XAI)")
    print("  [9] 2D Surrogate Decision Boundary (Toy Visualization)")
    print("  [A] All of the above (Default)")
    
    choice = input("\nEnter your choices (e.g., 1,C,R or A) [A]: ").strip().upper()
    if not choice or choice == "A":
        choices = ['0', 'R', 'C', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:
        choices = [c.strip() for c in choice.split(',')]
        
    if '0' in choices:
        generate_baseline_comparison()
        
    if 'R' in choices:
        generate_radar_chart()
        
    if 'C' in choices:
        generate_combined_roc_pr()
        
    if '1' in choices:
        generate_xgboost_importance(args.model)
        
    if '7' in choices:
        generate_smote_pca(args.model)
        
    if any(c in choices for c in ['2', '3', '4', '5', '6', '8', '9']):
        generate_inference_based_graphs(choices, args.model)
        
    print("\n[+] Selected mathematical representations successfully rendered to disk.")
    print("="*60)
