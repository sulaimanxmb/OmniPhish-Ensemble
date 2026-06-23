# Hyper-Detailed Architectural Comparison (CNN vs GNN)

This table contains advanced statistical rates derived directly from the confusion matrices of the Zero-Day vault evaluation.

| Statistical Metric | OmniPhish CNN | OmniPhish GNN | Difference | Winner |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 94.49% | 93.99% | 0.50% | 🏆 **CNN** |
| **Precision** | 92.83% | 95.45% | 2.62% | 🏆 **GNN** |
| **Recall (Sensitivity)** | 100.00% | 96.43% | 3.57% | 🏆 **CNN** |
| **Specificity (TNR)** | 80.79% | 87.20% | 6.42% | 🏆 **GNN** |
| **False Positive Rate** | 19.21% | 12.80% | 6.42% | 🏆 **GNN** |
| **False Negative Rate (Miss Rate)** | 0.00% | 3.57% | 3.57% | 🏆 **CNN** |
| **False Discovery Rate** | 7.17% | 4.55% | 2.62% | 🏆 **GNN** |
| **F1-Score** | 96.28% | 95.94% | 0.34% | 🏆 **CNN** |
| **MCC (Correlation)** | 0.8660 | 0.8442 | 0.0218 | 🏆 **CNN** |
| **Latency (ms/URL)** | 93.79 ms | 95.28 ms | 1.49 ms | 🏆 **CNN** |
| **Peak VRAM (MB)** | 1798.29 MB | 2964.40 MB | 1166.11 MB | 🏆 **CNN** |
