# Project Context

## 1. Overview
The AICS-Project-3 is a state-of-the-art **Zero-Day Phishing Detection System**. Its primary purpose is to identify highly evasive, visually-cloned phishing pages and Typosquatting attacks by ignoring visual rendering and instead analyzing the mathematical structure, semantic code intent, and routing infrastructure of web pages.

## 2. Architecture
The system employs a **Stacking Ensemble Architecture** that fuses three distinct deep learning and heuristic modalities into a unified prediction.

### Key Components & Data Flow:
1. **Data Ingestion Engine (`benign_manual_scrapper.py`, `phish_scraper.py`)**: Fetches post-rendered DOMs to bypass CAPTCHAs and obfuscation.
2. **Sanitation Engine (`cleaner.py`)**: Structurally validates inputs, strips noise, and removes MD5 duplicates.
3. **Feature Extraction (`dataset_loader.py`)**: Parses the raw HTML into three modalities:
   - **Modality 1 (Structural)**: 1D CNN processes raw HTML tags to identify anomalous structural nesting (128-D vector).
   - **Modality 2 (Semantic)**: CodeBERT processes JS/HTML snippets using an Overlapping Chunking Mechanism (to defeat token padding) and Global Max Pooling (768-D vector). Fine-tuned via PEFT/LoRA.
   - **Modality 3 (Lexical/Routing)**: Extracts hard heuristics like Levenshtein Brand Distance, DOM Depth, and Suspicious TLDs.
4. **Meta-Classifier (`trainer.py`, `classifier.py`)**: An XGBoost classifier ingests the concatenated 898-D hybrid vector. Optuna optimizes the XGBoost parameters on a strictly isolated validation subset.

## 3. Tech Stack
- **Language**: Python 3.x
- **Deep Learning / NLP**: PyTorch, HuggingFace Transformers (CodeBERT), PEFT (LoRA)
- **Machine Learning**: XGBoost, Scikit-Learn, Imbalanced-Learn (SMOTE), Optuna
- **Web Scraping / Parsing**: Playwright (Async), BeautifulSoup4
- **Hardware Optimization**: Apple Silicon MPS support configured for PyTorch

## 4. Key Features
- **Visual Evasion Resistance**: Bypasses image-based cloaking by operating exclusively on code structure and intent.
- **Overlapping Chunking**: Defeats sequence length limits, enabling CodeBERT to catch payloads hidden at the bottom of massive HTML files.
- **Domain Blindness Elimination**: Detects typosquatting via Levenshtein brand matching and raw IP/TLD flagging.
- **"Cyborg" Scraping**: Uses human-in-the-loop scraping on Top 1 Million domains to bypass enterprise WAFs and gather perfectly clean benign datasets.
- **Strict Optuna Validation**: Implements a zero-leak Optuna validation loop by quarantining 20% of training data before CNN feature extraction.

## 5. Current State
- **Completed Components**: 
  - CNN1D and CodeBERT Extraction Pipeline
  - XGBoost Meta-Classifier and K-Fold Evaluation Engine
  - Optuna Leak Fix implementation
  - DOM Depth Heuristics integration
  - CodeBERT LoRA fine-tuning architecture
  - Multi-directional Cyborg Scraper logic
  - Decoupled Zero-Day Evaluation Pipeline (`Check_for_overfitting.py`)
- **In-progress Work**: 
  - Mass Dataset Collection (Target: 5,000+ Samples).

## 6. Interfaces & Integrations
- **Data Sources**: Tranco Top 1 Million List (for benign domains), PhishTank/OpenPhish (for phishing URLs).
- **APIs**: HuggingFace Model Hub (for downloading base CodeBERT weights).

## 7. Security Considerations
- **Execution Sandbox**: The pipeline ingests live, potentially destructive zero-day JavaScript. Playwright handles execution in isolated browser contexts, but the downloaded raw HTML must never be executed outside of the sandboxed parser (`BeautifulSoup`).
- **Data Leakage Risk**: Strict K-Fold isolation protocols are enforced to ensure testing splits and Optuna validation splits never bleed into training processes, particularly prior to SMOTE balancing.

## 8. Known Issues / Limitations
- **Image-Based Phishing Blindspot**: The system is completely blind to visual pixels. If an attacker hosts a pure `.jpg` image of a login page with transparent HTML inputs, the structural and semantic engines will not flag it natively without relying heavily on the routing heuristics.
- **Hardware Bottlenecks**: Processing ~5,000 CodeBERT sequences is extremely computationally expensive. The pipeline is currently forced to use Phase-1 Global Extraction ("Fast Mode") to maintain reasonable execution times.

## 9. Next Steps
1. Complete manual "Cyborg" dataset collection to reach the 5,000+ sample threshold.
2. Execute the final Full-Pipeline training run (`trainer.py`).
3. Run `Check_for_overfitting.py` on the trained weights to generate the final Zero-Day IEEE statistics and confusion matrix.
