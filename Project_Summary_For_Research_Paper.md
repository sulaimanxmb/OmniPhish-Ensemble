# Dual-Modal Phishing Detection Ensemble: Eliminating Domain Blindness via Structural, Semantic, and Lexical Fusion

## 1. Abstract
The rapid proliferation of highly sophisticated zero-day phishing kits—specifically 1:1 visual clones and Single Page Applications (SPAs)—has rendered traditional visual-based detection mechanisms obsolete. Furthermore, relying solely on content analysis introduces "Domain Blindness," allowing attackers to bypass systems using typosquatted domains. This paper presents a novel **Stacking Ensemble Architecture** that eliminates Domain Blindness by fusing three distinct modalities: **DOM Structural Sequence Analysis (CNN1D)**, **Semantic Code Context Analysis (CodeBERT)**, and **Lexical URL Heuristics (Levenshtein Distance & TLD Analysis)**. By completely bypassing visual rendering and operating directly on raw DOM routing and lexicons, our system generates an 898-dimensional vector that is classified by an XGBoost Meta-Classifier, drastically outperforming traditional heuristic baselines.

## 2. Introduction
Modern phishing attacks heavily employ evasion techniques like CAPTCHA blockades, JavaScript-obfuscated rendering, and homograph typosquatting (e.g., `rnicrosoft.com` vs. `microsoft.com`). Visual similarity algorithms (like Siamese Networks) fail when an attacker uses a perfectly cloned enterprise login page. To counter this, our research shifts the detection paradigm from *how the page looks* to *how the page functions mathematically and structurally*. 

## 3. Methodology & System Architecture

### 3.1. Data Ingestion Pipeline
To capture the true state of zero-day phishing kits, the system utilizes an automated scraping engine built on Playwright with Firefox engine spoofing. This approach actively renders JavaScript, bypassing Cloudflare and anti-bot systems, to extract the post-rendered HTML DOM. 
- **Benign Data:** Collected via tiered scrapers, including an advanced anti-bot scraper and a "Cyborg" manual scraper designed to safely navigate multi-factor authentication (MFA) and Single Sign-On (SSO) enterprise logins.
- **Sanitation Engine:** Neural networks are highly sensitive to noise. The `cleaner.py` module structurally validates every HTML file, enforces the existence of credential `<input>` fields, and eliminates MD5 hash duplicates, ensuring a pristine dataset.

### 3.2. Feature Engineering & The Three Modalities

The core innovation is the fusion of three distinct analytical engines into a single predictive pipeline.

#### Modality 1: Structural Sequence Analysis (CNN1D)
Phishing kits often use automated templates that share a hidden mathematical structure, regardless of the visual CSS. The raw HTML is parsed into a sequence of structural tags, which is fed into a 1-Dimensional Convolutional Neural Network (CNN). The CNN identifies anomalous nested hierarchies and structural irregularities, outputting a **128-dimensional structural vector**.

#### Modality 2: Semantic Intent Analysis (CodeBERT)
To understand the *intent* of the malicious JavaScript and form submissions, a pre-trained CodeBERT transformer model is employed. Because transformers have a strict 512-token limit, we engineered an **HTML Distillation Engine** (`html_parser.py`) that explicitly strips visual noise (`<div>`, `<svg>`, `<style>`) and isolates only the semantic core: `<script>`, `<form>`, and `<a>` tags. CodeBERT compresses this contextual intent into a dense **768-dimensional semantic vector**.

#### Modality 3: Lexical URL & Routing Heuristics
To eliminate Domain Blindness, the system evaluates the target endpoint's location and routing behavior:
1. **Levenshtein Brand Distance:** Calculates the edit distance between the URL and Top 15 highly-phished enterprise brands, dynamically scaling suspicion for typosquatting (e.g., distance 1 or 2).
2. **Suspicious Domain Flagging:** Flags the use of raw IP addresses and low-cost, heavily abused Top-Level Domains (TLDs like `.xyz`, `.top`).
3. **Form Action Routing:** Scans `<form>` tags for suspicious data-drop endpoints (e.g., routing to a random `.php` script).
These heuristics produce a highly determinative scalar vector.

### 3.3. The Stacking Ensemble (XGBoost Meta-Classifier)
The outputs of the three modalities are flattened and concatenated into an **898-dimensional unified vector** (128 + 768 + 2 Lexical Heuristics). This vector is fed into an XGBoost Meta-Classifier.
- **Optimization Note:** The XGBoost model is specifically constrained (`max_depth=3`) to prevent overfitting on smaller datasets. Additionally, `tree_method='hist'` and thread limitations are employed to ensure stable execution on modern ARM architectures (e.g., Apple Silicon M-Series), preventing OpenMP segmentation faults.

### 3.4. Experimental Reproducibility & Deterministic Training
To guarantee exact statistical reproducibility for IEEE reporting, all stochastic components across the pipeline are strictly locked. A global random seed (`42`) is enforced across Python's native `random` module, NumPy, and PyTorch (including CPU, CUDA, and Apple MPS backends). Furthermore, non-deterministic PyTorch algorithms (such as cuDNN benchmark heuristics) are disabled. This ensures that the random data split, CNN weight initialization, and batch shuffling remain 100% deterministic, allowing independent verification of precision, recall, and F1-score metrics.

## 4. Model Training & Live Inference Workflow
1. **Baseline Comparison:** The architecture is evaluated against a control group—a Random Forest classifier relying purely on 8 manual HTML heuristics.
2. **Deep Learning Ensemble:** The core system trains the CNN on structural sequences, freezes CodeBERT to extract the 768-D vectors, and finally trains the XGBoost Stacking Classifier.
3. **Live Prediction:** During live inference (`predict.py`), the system parses a target URL, renders the DOM, extracts the three modalities in real-time, and provides both a confidence score and a detailed AI reasoning block explaining the structural vs. semantic rationale.

## 5. Conclusion
By fusing structural HTML sequences, distilled semantic JavaScript intent, and lexical typosquatting heuristics, the Dual-Modal Stacking Ensemble successfully identifies highly-evasive, visually-cloned zero-day phishing pages without relying on easily manipulated rendering algorithms. This architecture demonstrates that focusing on the intrinsic function and routing logic of a web page yields significantly higher precision and eliminates the Domain Blindness inherent in traditional visual scanners.
