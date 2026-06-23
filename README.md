# OmniPhish GNN Stacking Ensemble (Graph-Based Phishing Detection)

This repository contains the experimental implementation of the **OmniPhish Graph Neural Network (GNN) Stacking Ensemble**. While the 1D-CNN (on the `main` branch) is the definitive architecture for maximum Recall, this GNN architecture was designed to explore mathematical Graph Theory for maximizing Specificity and Precision.

### Architecture Overview (cuda-with-gnn branch)
- **Semantic Engine:** Microsoft CodeBERT (Fine-tuned via Low-Rank Adaptation - LoRA)
- **Structural Engine:** PyTorch Geometric (Graph Convolutional Network) processing the HTML Document Object Model (DOM) as a mathematical node-edge graph.
- **Meta-Classifier:** XGBoost (Gradient Boosting) processing the concatenated latent space.

*Note: For the official 1D-CNN architecture (which achieved 99.66% Recall), please switch back to the `main` branch.*

---

## 🚀 How to Run the Pipeline

The entire system is completely automated via the master pipeline script.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Ensure you have PyTorch Geometric installed correctly for your CUDA version.*

2. **Execute the Master Pipeline:**
   ```bash
   python run_pipeline.py
   ```
   *The pipeline will automatically prompt you to choose your training mode (Fast vs. Strict Isolation Slow Mode) and select which mathematical visualizations you want to generate.*

---

## 📂 Project Structure

```
OmniPhish-Ensemble/
├── run_pipeline.py            # The master execution script
├── requirements.txt           # Python dependencies (Includes torch_geometric)
├── omniphish/                 # Core framework modules
│   ├── dataset_loader.py      # TF-IDF, Heuristics, & DOM Graph extraction
│   ├── transformer_model.py   # CodeBERT + LoRA Architecture
│   └── classifier.py          # PyTorch Geometric GCN structural extractor
├── scripts/
│   ├── training/              # GNN Training loops and Vault evaluation
│   └── visualization/         # Plot generation (SHAP, t-SNE, PR Curves)
├── baselines/                 # SOTA Comparison Scripts
├── metrics/                   # CSV logs of K-Fold variance tests
├── pipeline_logs/             # Raw stdout logs from all scripts
├── visualizations/            # Automatically generated high-res PNGs
└── weights_gnn/               # (Manual Backup) Preserved trained model files
```

---

## 📊 Final IEEE Performance Metrics (Zero-Day Vault)
Evaluated on a strictly isolated 10% Zero-Day holdout vault (799 URLs) entirely unseen during the K-Fold training phase. 

Because graph theory extracts extremely complex spatial relationships, this architecture achieved superior Specificity and Precision compared to the CNN, minimizing false alarms.

- **Precision:** 95.45% *(Superior False Alarm Rejection)*
- **Recall (Sensitivity):** 96.43%
- **F1-Score:** 95.94%
- **Accuracy:** 93.99%
- **Inference Latency:** 95.28 ms / URL

## 📝 Branch Switching Note
If you switch back to the `main` branch to test the official CNN architecture, remember to manually rename `weights_gnn/` back to `weights/` when you return to this branch to avoid having to retrain the CodeBERT and GNN models from scratch.
