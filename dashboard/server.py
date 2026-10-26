import os
import sys
import json
import csv
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(
    title="OmniPhish Research Dashboard",
    description="Interactive evaluation and comparison dashboard for OmniPhish IEEE Conference Research Paper.",
    version="1.0.0"
)

# Root directory of OmniPhish project
ROOT_DIR = Path(__file__).resolve().parent.parent
VIS_DIR = ROOT_DIR / "visualizations"
METRICS_DIR = ROOT_DIR / "metrics"
LOGS_DIR = ROOT_DIR / "pipeline_logs"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

if VIS_DIR.exists():
    app.mount("/visualizations", StaticFiles(directory=str(VIS_DIR)), name="visualizations")

def get_canonical_metrics():
    """Default authoritative metrics from the locked paper configuration."""
    return {
        "sota_baselines": [
            {
                "model": "OmniPhish CNN (Proposed)",
                "category": "Proposed Architecture",
                "accuracy": 98.37,
                "precision": 98.25,
                "recall": 99.47,
                "f1": 98.86,
                "latency_ms": 93.79,
                "vram_gb": 1.80,
                "missed_attacks": 3,
                "total_attacks": 566,
                "highlight": True,
                "badge": "Best Recall & F1"
            },
            {
                "model": "OmniPhish GNN (Proposed)",
                "category": "Proposed Architecture",
                "accuracy": 93.24,
                "precision": 93.10,
                "recall": 97.70,
                "f1": 95.34,
                "latency_ms": 95.28,
                "vram_gb": 2.96,
                "missed_attacks": 13,
                "total_attacks": 566,
                "highlight": False,
                "badge": "High Stability"
            },
            {
                "model": "Long-Context Semantic (Longformer)",
                "category": "SOTA Baseline",
                "accuracy": 94.24,
                "precision": 94.07,
                "recall": 98.06,
                "f1": 96.02,
                "latency_ms": 412.30,
                "vram_gb": 11.56,
                "missed_attacks": 11,
                "total_attacks": 566,
                "highlight": False,
                "badge": "High VRAM (11.5GB)"
            },
            {
                "model": "Pure Structural CNN (HTMLPhish)",
                "category": "SOTA Baseline",
                "accuracy": 91.24,
                "precision": 92.03,
                "recall": 95.94,
                "f1": 93.94,
                "latency_ms": 35.00,
                "vram_gb": 0.64,
                "missed_attacks": 23,
                "total_attacks": 566,
                "highlight": False,
                "badge": "Lightweight (Blind to JS)"
            },
            {
                "model": "LLM Zero-Shot (Qwen2.5-Coder:14B)",
                "category": "Foundational Baseline",
                "accuracy": 47.68,
                "precision": 94.58,
                "recall": 27.74,
                "f1": 42.90,
                "latency_ms": 3500.00,
                "vram_gb": 9.50,
                "missed_attacks": 409,
                "total_attacks": 566,
                "highlight": False,
                "badge": "Severe Hallucinations"
            }
        ],
        "kfold_cv": {
            "cnn": {
                "accuracy": "93.78% ± 3.27%",
                "precision": "98.21% ± 2.33%",
                "recall": "93.50% ± 4.67%",
                "f1": "95.69% ± 2.10%",
                "roc_auc": "0.9425 ± 0.0281",
                "mcc": "0.8525 ± 0.0812",
                "fpr": "5.00% ± 6.67%"
            },
            "gnn": {
                "accuracy": "94.13% ± 0.42%",
                "precision": "96.29% ± 0.29%",
                "recall": "95.51% ± 0.59%",
                "f1": "95.90% ± 0.30%",
                "roc_auc": "0.9306 ± 0.0041",
                "mcc": "0.8561 ± 0.0096",
                "fpr": "9.38% ± 0.78%"
            }
        },
        "ablations": {
            "cnn": [
                {"name": "No CodeBERT (CNN + Heuristics)", "f1": 99.21, "precision": 99.47, "recall": 98.95, "mcc": 0.9727, "desc": "Spatial features alone; lacks semantic precision filtering."},
                {"name": "No CNN (CodeBERT + Heuristics)", "f1": 96.25, "precision": 97.82, "recall": 94.73, "mcc": 0.8760, "desc": "Missing spatial layout leads to 3% F1 drop."},
                {"name": "No Heuristics (CNN + CodeBERT)", "f1": 99.21, "precision": 99.65, "recall": 98.77, "mcc": 0.9728, "desc": "Slight loss in edge-case detection."},
                {"name": "Heuristics Only (No Deep Learning)", "f1": 86.55, "precision": 94.60, "recall": 79.76, "mcc": 0.6025, "desc": "Handcrafted rules fail against modern obfuscated kits."},
                {"name": "FULL CNN ENSEMBLE (All 903-D)", "f1": 99.21, "precision": 99.65, "recall": 98.77, "mcc": 0.9728, "desc": "Tri-Modal Synergy: Maximizes precision to 99.65%."}
            ],
            "gnn": [
                {"name": "No CodeBERT (GNN + Heuristics)", "f1": 88.95, "precision": 90.96, "recall": 87.02, "mcc": 0.6240, "desc": "GNN topology alone is vulnerable without JS intent."},
                {"name": "No GNN (CodeBERT + Heuristics)", "f1": 95.91, "precision": 96.50, "recall": 95.33, "mcc": 0.8546, "desc": "Removing GNN increases F1 (over-smoothing effect)."},
                {"name": "No Heuristics (GNN + CodeBERT)", "f1": 95.40, "precision": 95.65, "recall": 95.16, "mcc": 0.8350, "desc": "Heuristics act as geometric boundary regularizers."},
                {"name": "Heuristics Only (No Deep Learning)", "f1": 85.39, "precision": 91.65, "recall": 79.93, "mcc": 0.5864, "desc": "Fails against modern SPAs."},
                {"name": "FULL GNN ENSEMBLE (All 903-D)", "f1": 95.40, "precision": 95.81, "recall": 94.98, "mcc": 0.8355, "desc": "Demonstrates GNN over-smoothing in dense DOM trees."}
            ]
        },
        "vault_info": {
            "total_samples": 799,
            "phishing_samples": 566,
            "benign_samples": 233,
            "imbalance_ratio": "2.43 : 1",
            "isolation_method": "Strict Stratified Holdout Decoupling (10%)"
        }
    }

def load_kfold_csv():
    """Loads fold-by-fold results from metrics/kfold_variance_logs.csv if present."""
    csv_path = METRICS_DIR / "kfold_variance_logs.csv"
    if not csv_path.exists():
        return []
    
    rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "fold": int(row.get("Fold", 0)),
                    "train_f1": float(row.get("Train_F1", 0.0)) * 100,
                    "train_prec": float(row.get("Train_Precision", 0.0)) * 100,
                    "val_f1": float(row.get("Val_F1", 0.0)) * 100,
                    "val_prec": float(row.get("Val_Precision", 0.0)) * 100
                })
    except Exception as e:
        print(f"[!] Warning reading kfold csv: {e}")
    return rows

def get_available_visualizations():
    """Scans visualizations directory for PDF and image plots."""
    if not VIS_DIR.exists():
        return []
    
    vis_files = []
    for f in VIS_DIR.iterdir():
        if f.suffix.lower() in [".pdf", ".png", ".jpg", ".svg"]:
            vis_files.append({
                "name": f.name,
                "stem": f.stem.replace("_", " ").title(),
                "ext": f.suffix.lower(),
                "url": f"/visualizations/{f.name}"
            })
    return sorted(vis_files, key=lambda x: x["name"])

@app.get("/api/metrics")
def get_metrics_endpoint():
    data = get_canonical_metrics()
    data["kfold_folds"] = load_kfold_csv()
    data["visualizations"] = get_available_visualizations()
    return data

@app.get("/", response_class=HTMLResponse)
def index_page():
    template_file = TEMPLATES_DIR / "index.html"
    if not template_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard template not found")
    return template_file.read_text(encoding="utf-8")

def main():
    port = 5067
    host = "0.0.0.0"
    print("\n" + "="*70)
    print("🚀 OMNIPHISH IEEE RESEARCH WEB DASHBOARD")
    print("="*70)
    print(f"[*] Serving locally at:  http://localhost:{port}")
    print(f"[*] Network accessible:  http://127.0.0.1:{port}")
    print("="*70 + "\n")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
