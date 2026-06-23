# OmniPhish CNN Stacking Ensemble (Zero-Day Phishing Detection)

This repository contains the official implementation of the **OmniPhish 1D-CNN Stacking Ensemble**, designed for highly evasive, zero-day phishing URL detection. This architecture mathematically outperformed traditional ML baselines by fusing Deep Semantic representations (CodeBERT) with Structural Sequence modeling (1D-CNN) and explicit heuristic feature extraction.

### Architecture Overview (main branch)
- **Semantic Engine:** Microsoft CodeBERT (Fine-tuned via Low-Rank Adaptation - LoRA)
- **Structural Engine:** 1D Convolutional Neural Network (1D-CNN) processing raw HTML sequences
- **Meta-Classifier:** XGBoost (Gradient Boosting) processing the concatenated 899-dimensional latent space.

*Note: For the experimental Graph Neural Network (GNN) implementation, please switch to the `cuda-with-gnn` branch.*

---

## 🚀 How to Run the Pipeline

The entire system is completely automated via the master pipeline script.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

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
├── requirements.txt           # Python dependencies
├── omniphish/                 # Core framework modules
│   ├── dataset_loader.py      # TF-IDF, Heuristics, & Tokenization
│   ├── transformer_model.py   # CodeBERT + LoRA Architecture
│   └── classifier.py          # 1D-CNN structural extractor
├── scripts/
│   ├── training/              # CNN Training loops and Vault evaluation
│   └── visualization/         # Plot generation (SHAP, t-SNE, PR Curves)
├── baselines/                 # SOTA Comparison Scripts (TF-IDF, RF, etc.)
├── metrics/                   # CSV logs of K-Fold variance tests
├── pipeline_logs/             # Raw stdout logs from all scripts
├── visualizations/            # Automatically generated high-res PNGs
└── weights_cnn/               # (Manual Backup) Preserved trained model files
```

---

## 📊 Final IEEE Performance Metrics (Zero-Day Vault)
Evaluated on a strictly isolated 10% Zero-Day holdout vault (799 URLs) entirely unseen during the K-Fold training phase.

- **Recall (Sensitivity):** 99.66% *(0 False Negatives)*
- **Precision:** 93.10%
- **F1-Score:** 96.27%
- **Accuracy:** 94.37%
- **Inference Latency:** 93.79 ms / URL

## 📝 Branch Switching Note
If you switch branches to test the GNN architecture (`cuda-with-gnn`), remember to manually rename `weights_cnn/` back to `weights/` when you return to this branch to avoid having to retrain the CodeBERT and CNN models from scratch.
