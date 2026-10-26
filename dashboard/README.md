# 🌐 OmniPhish Interactive Research Dashboard

A modern, high-performance web dashboard running locally on port **5067** for analyzing, comparing, and exporting metrics from the OmniPhish IEEE Conference Research Paper.

---

## 🚀 Quick Start

Run the dashboard directly with Python:
```bash
python dashboard/server.py
```

Then open your browser to:
👉 **[http://localhost:5067](http://localhost:5067)**

---

## 📊 Features

1. **SOTA Baseline Comparison (Table I)**:
   - Interactive comparative charts (Accuracy, Precision, Recall, F1).
   - Side-by-side benchmarking against Pure CNN (HTMLPhish), Longformer (4096), and LLM Zero-Shot (Qwen2.5-Coder:14B).
2. **5-Fold Cross Validation Breakdown (Section VI-A)**:
   - Generalization metrics ($Mean \pm Std$).
   - Interactive fold-by-fold curve dynamically loaded from `metrics/kfold_variance_logs.csv`.
3. **Architectural Ablation Studies (Table II)**:
   - Quantifies Tri-Modal Synergy for the CNN variant.
   - Graphically explains the GNN over-smoothing paradox in dense DOM trees.
4. **GPU Hardware & Latency Profiling (Table III)**:
   - Real-time latency (ms/URL) vs VRAM allocation bubble plot.
5. **Research Artifacts Gallery**:
   - Direct access to high-resolution (300 DPI) PDF and PNG plots generated in `visualizations/`.
6. **One-Click IEEE LaTeX Export**:
   - Ready-to-paste LaTeX tables for `Paper.tex`.
