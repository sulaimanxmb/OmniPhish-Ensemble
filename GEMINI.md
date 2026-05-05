# Gemini AI Context - Dual-Modal Phishing Detection Ensemble

This file provides critical context for Gemini AI instances working on this repository.

## 1. Project Architecture
This project implements a Stacking Ensemble Architecture to detect zero-day phishing kits using three modalities:
- **Structural (CNN1D)**: Analyzes HTML structural tag sequences (128-D vector).
- **Semantic (CodeBERT)**: Analyzes contextual intent of JavaScript and form inputs via an overlapping chunking distillation engine (768-D vector).
- **Lexical/Routing Heuristics**: Analyzes Levenshtein distance, domain flags, and suspicious form actions (2-D vector).

These are mathematically concatenated into an 898-dimensional vector, which is classified by an XGBoost Meta-Classifier (optimized via Optuna). To handle class imbalance without data leakage, SMOTE is applied strictly within the training fold of a 5-Fold Stratified Cross-Validation loop.

## 2. Key Rules & Constraints
- **CRITICAL RULE**: For every single change or development made to this project, you MUST:
  1. Update `Project_Summary_For_Research_Paper.md` to reflect the changes.
  2. Update `DECISIONS.md` ONLY if a meaningful new technical or architectural decision was made (follow its internal formatting rules; no fluff).
  3. Update `CONTEXT.md` to accurately reflect the current state, impacted features, and architecture (follow its internal rules; overwrite/replace outdated info, do not stack logs).
  4. Commit all changes using `git`.
- **Hardware Profile**: The environment runs on macOS (Apple Silicon M-Series). Ensure all machine learning code (especially PyTorch and XGBoost) handles MPS (Metal Performance Shaders) properly and uses thread limitations (like `tree_method='hist'`) to prevent OpenMP segmentation faults.

## 3. Core Components
- `html_parser.py`: Distillation engine for stripping visual noise and extracting semantic targets.
- `dataset_loader.py`: The PyTorch dataloader handling feature extraction.
- `cleaner.py`: Dataset sanitation engine for eliminating duplicates and dead pages.
- `trainer.py` / `classifier.py`: Deep learning ensemble and XGBoost meta-classifier training logic.
- `baseline_trainer.py` / `baseline_predict.py`: Control group utilizing traditional Random Forest heuristics.
- `predict.py`: Live inference engine for evaluating target URLs in real-time.
