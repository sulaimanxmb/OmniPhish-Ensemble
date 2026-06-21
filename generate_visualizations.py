import os
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

def save_visualization(filename):
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
from omniphish.transformer_model import CodeBERTEmbedding
from omniphish.classifier import MetaClassifier

def get_feature_names():
    return [f"CNN_Feat_{i}" for i in range(128)] + \
           [f"CodeBERT_Feat_{i}" for i in range(768)] + \
           ["Heur_DOM_Depth", "Heur_Suspicious_Action", "Heur_URL_Score"]

def generate_xgboost_importance():
    print("\n[*] Generating XGBoost Feature Importance Plot...")
    meta = MetaClassifier()
    try:
        meta.load("weights/meta_classifier.pkl")
    except Exception as e:
        print(f"[!] Failed to load Meta-Classifier: {e}")
        return

    meta.xgb_model.get_booster().feature_names = get_feature_names()

    plt.figure(figsize=(10, 8))
    xgb.plot_importance(meta.xgb_model, max_num_features=20, importance_type="weight", 
                        title="Top 20 Ensemble Features (XGBoost)", 
                        color="#1f77b4", grid=False)
    plt.tight_layout()
    save_visualization('xgboost_importance.png')

def generate_smote_pca():
    print("\n[*] Initializing SMOTE Synthetic Data Generator...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    dirs = {'phishing': 'dataset/raw_html/phishing', 'benign': 'dataset/raw_html/benign'}
    dataset = PhishingDataset(dirs, undersample_benign=False)
    
    benign_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    phish_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
    
    selected_benign = random.sample(benign_indices, min(50, len(benign_indices)))
    selected_phish = random.sample(phish_indices, min(250, len(phish_indices)))
    
    imbalanced_idx = selected_benign + selected_phish
    imbalanced_subset = Subset(dataset, imbalanced_idx)
    loader = DataLoader(imbalanced_subset, batch_size=16, shuffle=False, collate_fn=custom_collate)
    
    cnn = CNN1DEmbedding(embedding_dim=64, num_filters=128).to(device)
    cnn.load_state_dict(torch.load("weights/cnn_trained.pt", map_location=device))
    cnn.eval()
    
    codebert = CodeBERTEmbedding(use_lora=True).to(device)
    codebert.load_state_dict(torch.load("weights/codebert_trained.pt", map_location=device))
    codebert.eval()
    
    print(f"[*] Extracting Mathematical Embeddings for {len(imbalanced_idx)} highly imbalanced samples...")
    cnn_feats, cb_feats, heuristics, labels_list = [], [], [], []
    
    with torch.no_grad():
        for batch_dicts, labels in tqdm(loader, desc="SMOTE Extraction"):
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
    y_true = np.array(labels_list, dtype=int)
    
    if len(heuristics.shape) == 1:
        heuristics = heuristics.reshape(-1, 1)
        
    X_imbalanced = np.concatenate([cnn_feats, cb_feats, heuristics], axis=1)
    
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
    
    plt.suptitle('SMOTE Effectiveness in OmniPhish 899-D Spatial Memory', fontweight='bold', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_visualization('smote_pca_comparison.png')


def generate_inference_based_graphs(choices):
    print("\n[*] Initializing Zero-Day Vault Inference...")
    if not os.path.exists("weights/vault_indices.npy"):
        print("[!] Error: 'weights/vault_indices.npy' not found.")
        return
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    vault_idx = np.load("weights/vault_indices.npy")
    print(f"[*] Loaded {len(vault_idx)} isolated Zero-Day samples.")
    
    dirs = {'phishing': 'dataset/raw_html/phishing', 'benign': 'dataset/raw_html/benign'}
    dataset = PhishingDataset(dirs, undersample_benign=False)
    vault_subset = Subset(dataset, vault_idx)
    vault_loader = DataLoader(vault_subset, batch_size=16, shuffle=False, collate_fn=custom_collate)
    
    print("[*] Loading Neural Networks...")
    cnn = CNN1DEmbedding(embedding_dim=64, num_filters=128).to(device)
    cnn.load_state_dict(torch.load("weights/cnn_trained.pt", map_location=device))
    cnn.eval()
    
    codebert = CodeBERTEmbedding(use_lora=True).to(device)
    codebert.load_state_dict(torch.load("weights/codebert_trained.pt", map_location=device))
    codebert.eval()
    
    meta = MetaClassifier()
    meta.load("weights/meta_classifier.pkl")
    
    print("[*] Performing Forward Pass (Extracting 899-D Mathematical Embeddings)...")
    cnn_feats, cb_feats, heuristics, labels_list = [], [], [], []
    
    with torch.no_grad():
        for batch_dicts, labels in tqdm(vault_loader, desc="Inference Extraction"):
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
    y_true = np.array(labels_list, dtype=int)
    
    if len(heuristics.shape) == 1:
        heuristics = heuristics.reshape(-1, 1)
        
    X_vault = np.concatenate([cnn_feats, cb_feats, heuristics], axis=1)
    
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
        save_visualization('vault_confusion_matrix.png')

    # 3. PCA
    if '3' in choices:
        print("\n[*] Compressing 899 Dimensions -> 2 Dimensions via PCA...")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_vault)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap=plt.cm.coolwarm, alpha=0.6, edgecolors='w', s=50)
        plt.title('PCA Projection of 899-D OmniPhish Latent Space', fontweight='bold', pad=15)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        cbar = plt.colorbar(scatter, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Benign (0)', 'Phishing (1)'])
        plt.grid(True, alpha=0.3)
        save_visualization('pca_clusters.png')

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

if __name__ == "__main__":
    print("="*60)
    print("📸 OMNIPHISH VISUALIZATION SUITE")
    print("="*60)
    print("Select the graphs you want to generate (comma-separated):")
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
    
    choice = input("\nEnter your choices (e.g., 1,5,7 or A) [A]: ").strip().upper()
    if not choice or choice == "A":
        choices = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:
        choices = [c.strip() for c in choice.split(',')]
        
    if '1' in choices:
        generate_xgboost_importance()
        
    if '7' in choices:
        generate_smote_pca()
        
    if any(c in choices for c in ['2', '3', '4', '5', '6', '8', '9']):
        generate_inference_based_graphs(choices)
        
    print("\n[+] Selected mathematical representations successfully rendered to disk.")
    print("="*60)
