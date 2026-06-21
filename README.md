# OmniPhish: Tri-Modal Stacking Ensemble

**Status:** Completed ✅

This repository contains the codebase for OmniPhish, a novel Deep Learning Stacking Ensemble designed to detect evasive zero-day phishing kits by analyzing raw Document Object Model (DOM) structural routing and JavaScript semantics, completely bypassing visual rendering (mitigating "Domain Blindness").

## Architecture Highlights
- **Context-Aware Structural Sequence Analysis (CNN1D)**: Analyzes HTML tag structures as contextual sequences.
- **Isolated JavaScript Semantic Intent Analysis (CodeBERT)**: Employs 1D Global Max Pooling over isolated JavaScript chunks to detect obfuscated payloads without hitting standard transformer length limits.
- **DOM Behavioral Heuristics**: Incorporates structural depth and suspicious externalized form actions.
- **XGBoost Meta-Classifier**: Fuses the multi-modal features via Optuna-optimized hyper-parameters.
- **Latent Space Imbalance Handling**: Employs fold-isolated SMOTE with Standard Scaling strictly inside the training split to geometrically balance the 899-D space without data leakage.

## Hardware Branches
This repository is split into two specialized branches depending on your hardware environment:
- `main`: Designed for low-resource hardware (Mac M-Series, laptops, CPUs). Uses `batch_size=1` and single-core DataLoaders to prevent memory exhaustion and IPC crashes.
- `cuda-optimized`: Designed for high-performance NVIDIA RTX machines. Uses `batch_size=32`, `num_workers=4/8` DataLoaders, Automatic Mixed Precision (AMP), and GPU-accelerated XGBoost. It also includes specific bypasses for Windows Win32 MAX_PATH and Windows Defender IPC blockages.

## Project Structure
- `omniphish/`: Core modules including data loaders, parsers, and neural network architectures (CNN1D & CodeBERT).
- `dataset_generator/`: Playwright-based autonomous ingestion pipeline for extracting raw HTML from evasive live sites.
- `baselines/`: State-of-the-art comparative models (Hybrid ML, HTMLPhish, XF-PhishBERT, SpecularNet GNN).
- `visualizations/`: Generated output graphs including PCA SMOTE clusters, PR/ROC curves, Confusion Matrices, and XAI Feature Importance.
- `trainer.py`: The core execution pipeline handling 5-Fold Cross-Validation, Optuna optimization, and strictly decoupled holdout validation.
- `predict.py`: Inference engine for real-world deployment.
- `ablation_study.py`: Script to generate the semantic-structural paradox metrics.

## Dataset Setup
The raw HTML dataset used to train this model is hosted externally to keep this repository lightweight. Before running the training pipeline, you must acquire the dataset:

1. **Download**: Navigate to the [OmniPhish Dataset v1 on Kaggle](https://www.kaggle.com/datasets/sulaimaneksambi/omniphish-dataset-v1) and download the archive.
   - *Alternatively, use the Kaggle CLI:* `kaggle datasets download -d sulaimaneksambi/omniphish-dataset-v1`
2. **Extract**: Unzip the downloaded file and place the folders at the root of this project exactly like this:
   - `Dataset/raw_html/phishing/`
   - `Dataset/raw_html/benign/`

## How to Run
1. Install dependencies: `pip install -r requirements.txt` (and run `playwright install`)

### Option A: Fully Automated Master Pipeline (Recommended)
You can automatically execute the entire pipeline unattended, which will run the models, the evaluations, the baselines, and the visualizations sequentially. It will generate a Master Dashboard of metrics at the end:
```bash
python run_pipeline.py
```

### Option B: Manual Execution
If you wish to run scripts individually:
1. Execute the main pipeline: `python trainer.py`
2. Check for overfitting on the isolated zero-day vault: `python Check_for_overfitting.py`
3. Generate analytical graphs: `python generate_visualizations.py`

*Note: This repository is currently anonymized for double-blind peer review.*
