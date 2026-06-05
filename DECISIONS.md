# Engineering Decisions

## Decision: Dual-Modal Stacking Ensemble (CNN1D + CodeBERT + XGBoost)

### Context
- Visual-based phishing detection mechanisms are vulnerable to sophisticated zero-day 1:1 clones and Single Page Applications (SPAs).
- Relying solely on textual heuristics causes "Domain Blindness" (vulnerability to typosquatting).

### Decision
- Shift from visual analysis to structural and functional analysis using a Stacking Ensemble architecture. 
- The pipeline fuses three distinct modalities: DOM Structural Sequence Analysis (CNN1D), Semantic Code Context Analysis (CodeBERT), and Lexical URL Heuristics (Levenshtein Distance & TLD Analysis) mapped into an XGBoost Meta-Classifier.

### Alternatives Considered
- Visual similarity models (Siamese Networks). Rejected due to vulnerability to perfect visual clones built on fundamentally different codebases.
- Pure Random Forest heuristic models. Rejected due to insufficient capability to map complex code interactions without deep learning.

### Trade-offs
- **Pros:** Highly robust against visual evasion; deeply analyzes attacker intent and routing infrastructure; highly accurate on zero-day kits.
- **Cons:** Significantly higher computational overhead for inference and training compared to simple heuristic models.

### Impact
- Establishes a highly resilient, state-of-the-art detection framework, shifting the computational burden to pre-extraction phases.

---

## Decision: Overlapping Chunking Mechanism for CodeBERT

### Context
- Standard transformer models like CodeBERT impose a strict 512-token limit.
- Attackers exploit this by padding payloads with massive blocks of whitespace or dead code, pushing malicious scripts out of the observable sequence.

### Decision
- Implemented an Overlapping Chunking Mechanism that divides the distilled HTML/JS sequence into 512-token windows with a 50-token overlap.
- Each chunk is processed independently, and a Global Max Pooling operation (`torch.max(dim=0)`) condenses the outputs into a single 768-D vector.

### Alternatives Considered
- Truncating sequences after 512 tokens. Rejected as it guarantees failure against padded evasion attacks.
- Using sparse transformers (e.g., Longformer). Rejected due to hardware memory limits and lack of code-specific pre-training compared to CodeBERT.

### Trade-offs
- **Pros:** Guarantees that malicious payloads split across chunk boundaries are not missed.
- **Cons:** Increases processing time linearly with the length of the document.

### Impact
- Eliminates padding-based evasion attacks, ensuring complete semantic coverage of massive JavaScript payloads.

---

## Decision: Strict Isolation of Optuna Validation Set

### Context
- During hyperparameter optimization, if the CNN is trained on the entire training fold, its extracted features for the validation split become artificially predictable (memorized).
- This "data leakage" caused Optuna to yield false F1-scores of 1.0, failing to properly tune the XGBoost meta-classifier.

### Decision
- Prior to CNN training, 20% of the active fold is strictly quarantined into an `Optuna_Val` subset.
- The CNN strictly trains only on the remaining 80% (`Sub_Train`). Optuna evaluates XGBoost purely on the `Optuna_Val` subset.

### Alternatives Considered
- Optimizing CNN and XGBoost simultaneously inside every Optuna trial. Rejected due to exponential, unfeasible training times (would take weeks on standard hardware).

### Trade-offs
- **Pros:** Restores mathematical validity to hyperparameter tuning, forcing Optuna to search for true optimal configurations.
- **Cons:** Reduces the volume of data the CNN sees during the hyperparameter tuning phase.

### Impact
- Prevents data leakage, ensuring the model's high accuracy metrics remain statistically valid for academic peer review (IEEE standards).

---

## Decision: CodeBERT PEFT/LoRA Fine-Tuning

### Context
- CodeBERT is a generalized code transformer (125M parameters). It lacks specialized vocabulary mapping for heavily obfuscated phishing payloads.
- Fine-tuning the entire model on a small dataset leads to severe catastrophic forgetting and overfitting.

### Decision
- Implemented Low-Rank Adaptation (LoRA) via the `peft` library.
- Injected trainable low-rank matrices into the "query" and "value" attention heads while freezing the base 125M weights.

### Alternatives Considered
- Full Fine-Tuning. Rejected due to extreme memory requirements and overfitting risks.
- Static Feature Extraction (Zero Fine-Tuning). Rejected as it failed to capture nuanced obfuscation patterns efficiently.

### Trade-offs
- **Pros:** Drastically increases feature extraction accuracy for phishing-specific vocabulary while keeping memory consumption low.
- **Cons:** Requires additional preprocessing time during Phase 1 training execution.

### Impact
- Allows dynamic adaptation to zero-day phishing kits without compromising the core knowledge base of the transformer.

---

## Decision: DOM Depth Heuristics for Structural Profiling

### Context
- Phishing obfuscators recursively generate massive numbers of redundant `<div>` and `<span>` tags.
- A purely sequential model (CNN1D) may struggle to explicitly quantify the absolute "messiness" of the code hierarchy.

### Decision
- Implemented a fast recursive DOM parser to calculate `Max_Depth` and `Average_Depth` of the HTML tree.
- These two integers were appended to the Lexical/Routing Heuristic array.

### Alternatives Considered
- Graph Neural Networks (GNNs). Acknowledged as technically superior, but rejected for the current scope due to massive complexity increases and hardware constraints.

### Trade-offs
- **Pros:** Provides a strict, hard mathematical proxy for obfuscation complexity that XGBoost easily interprets.
- **Cons:** Marginally increases the data ingestion time per URL.

### Impact
- Creates a powerful synergy between the CNN's spatial analysis and XGBoost's numerical profiling, heavily punishing automated templating systems.

---

## Decision: "Cyborg" Manual Scraper for Benign Dataset

### Context
- Automated scrapers face hard blocks (CAPTCHAs, Cloudflare) when accessing enterprise Single Sign-On (SSO) portals.
- Failing to access the actual post-login portals starves the benign dataset of complex, legitimate React/Angular enterprise code structures.

### Decision
- Engineered a "Cyborg" Manual Scraper (`benign_manual_scrapper.py`) using Playwright.
- The system automates URL traversal (Top 1 Million) but yields manual control to a human operator to bypass CAPTCHAs and navigate deep into portals before a custom UI injection button is clicked to save the post-rendered DOM.

### Alternatives Considered
- Purely automated headless scraping. Rejected due to overwhelming failure rates against modern enterprise WAFs.
- Purchasing pre-rendered datasets. Rejected due to staleness and lack of SPA execution.

### Trade-offs
- **Pros:** Guarantees absolute highest-quality, zero-noise data from actual secure enterprise environments.
- **Cons:** Extreme time investment required for dataset generation.

### Impact
- Elevates the dataset quality to academic research standards, completely eliminating noise-induced false positives during model training.

---

## Decision: Decoupled Evaluation Pipeline (Zero-Day Vault)

### Context
- The system must provide absolute proof of model generalization for IEEE publication via learning curve analysis and holdout testing.
- Performing data logging, PyTorch training, XGBoost training, SMOTE, Optuna, and `matplotlib` rendering inside a single `trainer.py` script frequently caused Out-of-Memory (OOM) crashes on 8GB Apple Silicon hardware.

### Decision
- Implemented a decoupled evaluation architecture. `trainer.py` isolates a 10% stratified "Zero-Day Vault" upfront, saves the indices to disk, and performs lightweight K-Fold variance CSV logging during training.
- A completely independent script, `Check_for_overfitting.py`, is executed after `trainer.py` shuts down (freeing all RAM). It reads the CSV logs to prove K-Fold variance stability and runs live inference on the 10% Zero-Day Vault to generate the final confusion matrix.

### Alternatives Considered
- Infusing all graph generation and holdout testing directly into `trainer.py`. Rejected due to heavy memory overhead and high risk of SSD swapping/crashes on 8GB Unified Memory configurations.

### Trade-offs
- **Pros:** Completely eliminates OOM risk by freeing PyTorch RAM before evaluation; provides mathematically strict proof of non-overfitting on zero-day datasets.
- **Cons:** Requires the user to execute two scripts sequentially instead of a single automated pipeline.

### Impact
- Guarantees training stability on lower-end hardware while generating pristine, unquestionable metrics for IEEE peer review.
