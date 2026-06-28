# FINAL METRICS (LOCKED SEEDS - SLOW MODE)
**Execution Date**: Pipeline Run
**Total Execution Time**: ~2.5 hours

## 1. K-FOLD CROSS VALIDATION (5-Fold, Slow Mode Zero-Leak)
**CNN Architecture**
- Average Accuracy:  93.78% ± 3.27%
- Average Precision: 98.21% ± 2.33%
- Average Recall:    93.50% ± 4.67%
- Average F1-Score:  95.69% ± 2.10%
- Average ROC-AUC:   0.9425 ± 0.0281
- Average MCC:       0.8525 ± 0.0812
- Average FPR:       5.00% ± 6.67%

**GNN Architecture**
- Average Accuracy:  94.13% ± 0.42%
- Average Precision: 96.29% ± 0.29%
- Average Recall:    95.51% ± 0.59%
- Average F1-Score:  95.90% ± 0.30%
- Average ROC-AUC:   0.9306 ± 0.0041
- Average MCC:       0.8561 ± 0.0096
- Average FPR:       9.38% ± 0.78%

## 2. ZERO-DAY VAULT INFERENCE (799 Decoupled Samples)
**OmniPhish CNN Variant**
- Accuracy:  98.37%
- Precision: 98.25%
- Recall:    99.47% (Missed exactly 3 out of 566 phishing attacks)
- F1-Score:  98.86%
- Latency:   94.60 ms per URL
- Peak RAM:  2991.46 MB

**OmniPhish GNN Variant**
- Accuracy:  93.24%
- Precision: 92.88%
- Recall:    97.63%
- F1-Score:  95.20%
- Latency:   98.93 ms per URL
- Peak RAM:  3026.31 MB

## 3. STATE-OF-THE-ART BASELINES (Zero-Day Vault)
**Pure Structural CNN (HTMLPhish)**
- Accuracy:  91.24%
- Precision: 92.06%
- Recall:    95.95%
- F1-Score:  93.97%
- ROC-AUC:   0.9751
- Peak VRAM: 0.64 GB

**Long-Context Semantic (Longformer 4096)**
- Accuracy:  94.24%
- Precision: 94.09%
- Recall:    98.06%
- F1-Score:  96.03%
- ROC-AUC:   0.9851
- Peak VRAM: 11.56 GB

**LLM Zero-Shot (Qwen2.5-Coder:14B)**
- Accuracy:  47.68%
- Precision: 94.64%
- Recall:    27.99% (Missed 409 attacks)
- F1-Score:  43.21%

## 4. ARCHITECTURAL ABLATION STUDY
### CNN Ablation Results
| Ablated Architecture | F1-Score | Precision | Recall | MCC |
|----------------------|----------|-----------|--------|-----|
| No CodeBERT (CNN+Heur) | 99.21% | 99.47% | 98.95% | 0.9727 |
| No CNN (CodeBERT+Heur) | 96.25% | 97.82% | 94.73% | 0.8760 |
| No Heur (CNN+CodeBERT) | 99.21% | 99.65% | 98.77% | 0.9728 |
| **FULL CNN ENSEMBLE**  | **99.21%** | **99.65%** | **98.77%** | **0.9728** |

### GNN Ablation Results
| Ablated Architecture | F1-Score | Precision | Recall | MCC |
|----------------------|----------|-----------|--------|-----|
| No CodeBERT (GNN+Heur) | 88.95% | 90.96% | 87.02% | 0.6240 |
| No GNN (CodeBERT+Heur) | 95.91% | 96.50% | 95.33% | 0.8546 |
| No Heur (GNN+CodeBERT) | 95.40% | 95.65% | 95.16% | 0.8350 |
| **FULL GNN ENSEMBLE**  | **95.40%** | **95.81%** | **94.98%** | **0.8355** |
