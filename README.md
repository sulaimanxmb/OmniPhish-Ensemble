# OmniPhish CNN Stacking Ensemble (Zero-Day Phishing Detection)

This repository contains the official implementation of the **OmniPhish 1D-CNN Stacking Ensemble**, designed for highly evasive, zero-day phishing URL detection. This architecture mathematically outperformed traditional ML baselines by fusing Deep Semantic representations (CodeBERT) with Structural Sequence modeling (1D-CNN) and explicit heuristic feature extraction.

### 🌐 Public Resources & Live Demo
- **Live Interactive Demo (Gradio):** [Hugging Face Spaces](https://huggingface.co/spaces/XMB480/Omniphish)
- **Pre-Trained Model Weights:** [Hugging Face Model Hub](https://huggingface.co/XMB480/OmniPhish-Ensemble)
- **Official Training Dataset:** [OmniPhish Dataset v1 on Kaggle](https://www.kaggle.com/datasets/sulaimaneksambi/omniphish-dataset-v1)

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

2. **Execute the Master Pipeline (Reproduce all metrics):**
   ```bash
   python run_pipeline.py
   ```
   *The pipeline will automatically prompt you to choose your training mode (Fast vs. Strict Isolation Slow Mode) and select which mathematical visualizations you want to generate. Once training is complete, the script will systematically test the ensemble against the isolated Zero-Day Vault and output the final Recall, Precision, F1-Score, and Accuracy metrics directly to your terminal. All raw metrics and generated PR/ROC curves will be automatically saved in the `metrics/` and `visualizations/` directories.*

### 🤖 Running Independent Baselines (For IEEE Paper)
To independently verify the baseline metrics without running the full OmniPhish pipeline, you can execute them individually. Note that these scripts output their results to the terminal and include **Peak VRAM** tracking.

1. **HTMLPhish (Character-Level CNN):**
   ```bash
   python baselines/htmlphish_trainer.py
   ```
2. **Longformer (Long-Context Transformer):**
   ```bash
   python baselines/longformer_trainer.py
   ```
3. **Qwen2.5-Coder:14B (Large Language Model via Ollama):**
   *Note: This specific script requires the Ollama engine to bypass PyTorch VRAM limits.*
   - Install the Windows client from: https://ollama.com/download/windows
   - Download the model in a background terminal: `ollama run qwen2.5-coder:14b`
   - Run the evaluation script:
   ```bash
   python baselines/llm_zeroshot_baseline.py
   ```

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
