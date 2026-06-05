# OmniPhish: Tri-Modal Stacking Ensemble

**Status:** Work in Progress 🚧

This repository contains the codebase for a novel Deep Learning Stacking Ensemble designed to detect zero-day phishing kits by analyzing raw Document Object Model (DOM) routing and JavaScript semantics, bypassing visual rendering.

## Architecture Highlights
- **Context-Aware Structural Sequence Analysis (CNN1D)**: Analyzes HTML tag structures as sequences.
- **Isolated JavaScript Semantic Intent Analysis (CodeBERT)**: Employs an attention-pooling layer to detect obfuscated JavaScript logic.
- **DOM Behavioral Heuristics**: Incorporates structural depth and suspicious form actions.
- **XGBoost Meta-Classifier**: Fuses the multi-modal features via Optuna-optimized hyper-parameters.

## Project Structure
- Data loaders and preprocessors for HTML/DOM extraction.
- Neural network weights and model architectures for the Tri-Modal Ensemble.
- Baseline models (TF-IDF, Random Forest, Raw CNN) for comparison.
- Pipeline for applying SMOTE strictly inside the training split to avoid data leakage.

*More documentation, results, and setup instructions will be added as the project nears finalization.*
