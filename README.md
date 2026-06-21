# OmniPhish: Tri-Modal Stacking Ensemble

**Status:** Completed ✅

This repository contains the codebase for OmniPhish, a novel Deep Learning Stacking Ensemble designed to detect evasive zero-day phishing kits by analyzing raw Document Object Model (DOM) structural routing and JavaScript semantics, completely bypassing visual rendering (mitigating "Domain Blindness").

## Architecture Highlights
- **Context-Aware Structural Sequence Analysis (CNN1D)**: Analyzes HTML tag structures as contextual sequences.
- **Isolated JavaScript Semantic Intent Analysis (CodeBERT)**: Employs 1D Global Max Pooling over isolated JavaScript chunks to detect obfuscated payloads without hitting standard transformer length limits.
- **DOM Behavioral Heuristics**: Incorporates structural depth and suspicious externalized form actions.
- **XGBoost Meta-Classifier**: Fuses the multi-modal features via Optuna-optimized hyper-parameters.
- **Latent Space Imbalance Handling**: Employs fold-isolated SMOTE with Standard Scaling strictly inside the training split to geometrically balance the 899-D space without data leakage.

## Project Structure
- `omniphish/`: Core modules including data loaders, parsers, and neural network architectures (CNN1D & CodeBERT).
- `dataset_generator/`: Playwright-based autonomous ingestion pipeline for extracting raw HTML from evasive live sites.
- `baselines/`: State-of-the-art comparative models (Hybrid ML, HTMLPhish, XF-PhishBERT, SpecularNet GNN).
- `visualizations/`: Generated output graphs including PCA SMOTE clusters, PR/ROC curves, Confusion Matrices, and XAI Feature Importance.
- `trainer.py`: The core execution pipeline handling 5-Fold Cross-Validation, Optuna optimization, and strictly decoupled holdout validation.
- `predict.py`: Inference engine for real-world deployment.
- `ablation_study.py`: Script to generate the semantic-structural paradox metrics.

## How to Run
1. Install dependencies: `pip install -r requirements.txt` (and run `playwright install`)
2. Execute the main pipeline: `python trainer.py`
3. Check for overfitting on the isolated zero-day vault: `python Check_for_overfitting.py`
4. Generate analytical graphs: `python generate_visualizations.py`

*Note: This repository is currently anonymized for double-blind peer review.*
