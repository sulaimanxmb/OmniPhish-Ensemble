# Zero-Day Phishing Stacking Ensemble - Quick Start Guide

This guide contains everything you need to recreate, build, and run the Dual-Modal Phishing Detection Ensemble from scratch in a fresh environment. 

## 1. Project Overview
This project is an advanced **Stacking Ensemble** designed to catch highly sophisticated zero-day phishing kits (including 1:1 visual clones and Single Page Applications). It completely bypasses visual rendering and analyzes the underlying DOM routing.

The architecture fuses three modalities:
1. **Structural (CNN1D):** Reads the mathematical sequence of the raw HTML (128 dimensions).
2. **Semantic (CodeBERT):** Reads the contextual intent of JavaScript and form inputs, compressed via a 512-token distillation engine (768 dimensions).
3. **Routing Heuristic:** Scans `<form>` tags for suspicious data-drop endpoints (1 dimension).

These combine into an **897-dimensional vector** that is classified by an **XGBoost Meta-Classifier**.

---

## 2. Environment Setup

**Prerequisites:** Python 3.9+ (Optimized for Apple Silicon / Mac M-Series).

1. **Clone/Create the Directory:**
   Create a fresh folder and move your python scripts into it.

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you run into PyTorch Mac-silicon errors, ensure you install the MPS-compatible torch version.*

3. **Install the Web Scraper Engines:**
   This project uses Playwright with Firefox engine spoofing to bypass Cloudflare and anti-bot systems.
   ```bash
   playwright install firefox
   ```

---

## 3. Dataset Generation (The Ingestion Pipeline)

The neural networks require raw, post-JavaScript rendered HTML files. The directory structure **must** look exactly like this:
```text
dataset/
└── raw_html/
    ├── benign/     # Safe enterprise login pages
    └── phishing/   # Zero-day phishing templates
```

### Populating Phishing Data
Run the autonomous scraper. It reads live malicious URLs (from OpenPhish or your `links.txt`) and downloads their fully-rendered HTML blocks.
```bash
python phish_scraper.py
```

### Populating Benign Data
We use three different scrapers depending on the complexity of the target enterprise sites:
1. **Standard Scraper:** `python benign_scrapper.py`
2. **Advanced Anti-Bot Scraper:** `python benign_advanced_scrapper.py`
3. **The "Cyborg" Manual Scraper:** `python benign_manual_scrapper.py` *(Use this for complex Microsoft/SSO logins. It launches a browser with a floating "SAVE" button so a human can bypass 2FA before capturing the DOM).*

### Dataset Sanitation
Neural networks will overfit if fed garbage data (e.g., 404 pages or CAPTCHA blockades). 
Run the cleaner to structurally validate every file, verify credential `<input>` fields exist, and delete MD5 duplicates.
```bash
python cleaner.py
```
*(Note: The PyTorch dataloader automatically caps the datasets symmetrically to prevent XGBoost bias).*

---

## 4. Model Training

Once your `dataset/raw_html/` folder is populated with `.html` files, you must train the models in order. The weights will be automatically saved into a new `weights/` directory.

### Step 1: Train the Baseline (Random Forest)
This trains a traditional machine learning model relying purely on 8 manual heuristics (HTML length, num scripts, suspicious form actions, etc.). This serves as the control group for your research metrics.
```bash
python baseline_trainer.py
```

### Step 2: Train the Deep Learning Ensemble
This is the core engine. It will automatically run through 3 distinct phases:
1. Train the CNN to understand structural tag sequences.
2. Freeze CodeBERT and extract the 897-dimensional feature vectors.
3. Train the XGBoost Stacking Classifier on those dense vectors.
```bash
python trainer.py
```

---

## 5. Live Prediction

Once the `weights/` folder contains `cnn_trained.pt` and `meta_classifier.pkl`, your system is armed and ready for live zero-day URL evaluation!

**To test the Deep Learning Ensemble:**
```bash
python predict.py "https://target-login-page.com"
```
*Output: Provides the confidence score and a detailed AI Reasoning block explaining exactly WHY the neural network classified the site as benign or malicious (Semantic vs Structural vs Routing).*

**To test the Baseline Model (for comparison):**
```bash
python baseline_predict.py "https://target-login-page.com"
```
*Output: Displays the exact 8 heuristic values (e.g., Num Scripts: 23, Suspicious Action: 1) used to make the decision.*

---

## 6. Core Component Glossary

If you need to edit the architecture in the new folder, here is what each core file does:
*   **`html_parser.py`**: The "Distillation Engine". Strips out visual noise (`<div>`, `<img>`) and extracts purely `<form>`, `<script>`, and `<a>` to fit CodeBERT's strict 512-token limit.
*   **`dataset_loader.py`**: The PyTorch dataloader. Parses filenames, extracts the Suspicious Form Action heuristic, and feeds the `(CNN, CodeBERT, Heuristic)` tuples to the trainer.
*   **`classifier.py`**: Holds the logic for the XGBoost Meta-Classifier and the mathematical concatenation of the final 897-dimensional vector.
*   **`baseline_features.py`**: Contains the regex and BeautifulSoup logic to extract the manual heuristics (like detecting `login.php` drops or raw IP addresses).
