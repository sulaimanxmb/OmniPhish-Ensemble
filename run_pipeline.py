import os
import sys
import io
import subprocess
import time
import re

# Prevent pipeline crashes if the user's internet connection drops
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def run_script(script_cmd, inject_input=False):
    script_display_name = script_cmd[0] if isinstance(script_cmd, list) else script_cmd
    print(f"\n{'='*60}\n🚀 EXECUTING: {script_display_name} {' '.join(script_cmd[1:]) if isinstance(script_cmd, list) else ''}\n{'='*60}")
    
    log_dir = "pipeline_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    safe_name = script_display_name.replace("/", "_").replace("\\", "_")
    if isinstance(script_cmd, list) and len(script_cmd) > 1:
        safe_name += "_" + "_".join(script_cmd[1:]).replace("-", "")
    log_file = os.path.join(log_dir, f"{safe_name}.log")
    
    cmd = [sys.executable] + script_cmd if isinstance(script_cmd, list) else [sys.executable, script_cmd]
    
    try:
        if inject_input:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            process.stdin.write(f"{inject_input}\n".encode('utf-8'))
            process.stdin.flush()
        else:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            
        stdout_wrapper = io.TextIOWrapper(process.stdout, encoding='utf-8', errors='ignore', newline='')
            
        full_log = []
        current_line = []
        with open(log_file, "w", encoding="utf-8") as f:
            while True:
                char = stdout_wrapper.read(1)
                if not char:
                    if current_line:
                        line = "".join(current_line)
                        f.write(line)
                        full_log.append(line)
                    break
                
                sys.stdout.write(char)
                sys.stdout.flush()
                
                if char == '\n':
                    line = "".join(current_line) + '\n'
                    f.write(line)
                    full_log.append(line)
                    current_line = []
                elif char != '\r':
                    current_line.append(char)
                
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[!] ERROR: {script_display_name} crashed with exit code {process.returncode}")
            return False, full_log
            
        return True, full_log
    except Exception as e:
        print(f"\n[!] Failed to execute {script_display_name}: {e}")
        return False, []

def extract_metrics(log_lines):
    """Scans the bottom of the log to extract common evaluation metrics."""
    metrics = []
    # Scan the last 150 lines
    for line in log_lines[-150:]:
        line_lower = line.lower()
        if "accuracy" in line_lower and ":" in line_lower:
            # Simple regex to grab the number
            match = re.search(r'accuracy[^\d]+([0-9]+\.[0-9]+)', line_lower)
            if match:
                metrics.append(f"Accuracy: {match.group(1)}")
                
        if "f1-score" in line_lower or "f1 score" in line_lower:
            if ":" in line_lower or "macro avg" in line_lower or "weighted avg" in line_lower:
                match = re.search(r'([0-9]+\.[0-9]+)', line_lower)
                if match:
                    metrics.append(line.strip())
                    
    # Deduplicate and format
    unique_metrics = list(dict.fromkeys(metrics))
    # Filter to most relevant 2-3 lines
    return unique_metrics[-3:] if unique_metrics else ["Completed successfully (No generic metrics extracted). Check logs."]

def check_dataset_integrity():
    """Validates that the exact expected number of raw HTML files exist to prevent corrupted runs."""
    benign_dir = os.path.join("Dataset", "raw_html", "benign")
    phish_dir = os.path.join("Dataset", "raw_html", "phishing")
    
    if not os.path.exists(benign_dir) or not os.path.exists(phish_dir):
        print("\n[!] FATAL ERROR: Dataset directories not found! Please extract Dataset.zip into the root folder.")
        sys.exit(1)
        
    benign_count = len([f for f in os.listdir(benign_dir) if f.endswith(".html")])
    phish_count = len([f for f in os.listdir(phish_dir) if f.endswith(".html")])
    
    EXPECTED_BENIGN = 2250
    EXPECTED_PHISH = 5740
    
    if benign_count != EXPECTED_BENIGN or phish_count != EXPECTED_PHISH:
        print("\n" + "!"*80)
        print("🚨 DATASET CORRUPTION DETECTED 🚨")
        print(f"Expected Benign: {EXPECTED_BENIGN} | Found: {benign_count}")
        print(f"Expected Phishing: {EXPECTED_PHISH} | Found: {phish_count}")
        print("Please re-extract the OmniPhish Kaggle dataset cleanly to prevent corrupted research metrics.")
        print("!"*80 + "\n")
        sys.exit(1)
        
    print(f"[*] Dataset Integrity Verified: {benign_count} Benign, {phish_count} Phishing.")

def main():
    check_dataset_integrity()
    start_time = time.time()
    
    print("\n" + "#"*80)
    print("OmniPhish - Master Research Pipeline Execution")
    print("#"*80)
    print("\nThis script will automatically execute the entire pipeline unattended.")
    print("All outputs will be mirrored to the 'pipeline_logs/' directory.\n")
    
    print("==================================================")
    print("🚀 PIPELINE CONFIGURATION")
    print("==================================================")
    print("[1] RUN BOTH CNN & GNN (Default)")
    print("[2] RUN CNN ONLY")
    print("[3] RUN GNN ONLY")
    print("[4] SKIP ALL TRAINING (Evaluations only)")
    print("==================================================")
    skip_choice = input("Enter 1, 2, 3, or 4 [Default: 1]: ").strip()
    
    print("\n==================================================")
    print("🎨 VISUALIZATIONS CONFIGURATION")
    print("==================================================")
    print("Select the graphs you want to generate (comma-separated):")
    print("  [1] XGBoost Feature Importance Plot")
    print("  [2] Confusion Matrix Heatmap")
    print("  [3] PCA Scatter Plot (899-D -> 2D)")
    print("  [4] t-SNE Scatter Plot (899-D -> 2D)")
    print("  [5] ROC Curve & AUC Score")
    print("  [6] Precision-Recall (PR) Curve")
    print("  [7] SMOTE Spatial Imbalance Demonstration")
    print("  [8] SHAP Game Theory Values (Expensive)")
    print("  [9] 2D Surrogate Decision Boundary")
    print("  [A] All of the above (Default)")
    print("  [NONE] Skip visualizations entirely")
    print("==================================================")
    vis_choice = input("Enter choices (e.g., 1,5,7 or A) [Default: A]: ").strip().upper()
    
    if vis_choice == "NONE":
        do_vis = False
    elif not vis_choice:
        vis_choice = "A"
        do_vis = True
    else:
        do_vis = True
    
    mode_choice = '1'
    if skip_choice == '4':
        pass # Skip training
    else:
        # Ask for Training Mode
        print("\n==================================================")
        print("🚀 SELECT TRAINING MODE FOR THE PIPELINE")
        print("==================================================")
        print("[1] FAST MODE (Global CodeBERT PEFT, Feature-Leak accepted, ~15 mins)")
        print("[2] SLOW MODE (Strict K-Fold CodeBERT Isolation, Zero-Leak, ~2 hours)")
        print("==================================================")
        mode_choice = input("Enter 1 or 2 [Default: 1]: ").strip()
        if mode_choice not in ['1', '2']:
            mode_choice = '1'
            
    # Build SCRIPTS_TO_RUN dynamically
    SCRIPTS_TO_RUN = []
    
    cnn_script = ("trainer.py" if os.path.exists("trainer.py") else "scripts/training/trainer.py", "PROMPT_MODE")
    gnn_script = ("gnn_trainer.py" if os.path.exists("gnn_trainer.py") else "scripts/training/gnn_trainer.py", "PROMPT_MODE")
    
    models_to_eval = []
    
    if skip_choice == '2':
        SCRIPTS_TO_RUN.append(cnn_script)
        models_to_eval.append("cnn")
    elif skip_choice == '3':
        SCRIPTS_TO_RUN.append(gnn_script)
        models_to_eval.append("gnn")
    elif skip_choice == '4':
        # Default to both for eval only unless specified
        models_to_eval = ["cnn", "gnn"]
    else: # Default 1: Both
        SCRIPTS_TO_RUN.append(cnn_script)
        SCRIPTS_TO_RUN.append(gnn_script)
        models_to_eval.append("cnn")
        models_to_eval.append("gnn")
        
    chk_script = "Check_for_overfitting.py" if os.path.exists("Check_for_overfitting.py") else "scripts/training/Check_for_overfitting.py"
    abl_script = "ablation_study.py" if os.path.exists("ablation_study.py") else "scripts/training/ablation_study.py"
    vis_script = "generate_visualizations.py" if os.path.exists("generate_visualizations.py") else "scripts/visualization/generate_visualizations.py"
    
    for model in models_to_eval:
        SCRIPTS_TO_RUN.append(([chk_script, "--model", model], False))
        SCRIPTS_TO_RUN.append(([abl_script, "--model", model], False))
        
    SCRIPTS_TO_RUN.append(("baselines/htmlphish_trainer.py", False))
    SCRIPTS_TO_RUN.append(("baselines/longformer_trainer.py", False))
    SCRIPTS_TO_RUN.append(("baselines/llm_zeroshot_baseline.py", False))
    
    if do_vis:
        for model in models_to_eval:
            SCRIPTS_TO_RUN.append(([vis_script, "--model", model], "VIS_PROMPT_MODE"))
    
    results_dashboard = {}
    
    for script_cmd, inject_input in SCRIPTS_TO_RUN:
        script_path = script_cmd[0] if isinstance(script_cmd, list) else script_cmd
        if not os.path.exists(script_path):
            print(f"[!] Warning: {script_path} not found in this branch! Skipping.")
            results_dashboard[str(script_cmd)] = ("SKIPPED", ["File not found"])
            continue
            
        # Resolve dynamic input
        if inject_input == "PROMPT_MODE":
            current_input = mode_choice
        elif inject_input == "VIS_PROMPT_MODE":
            current_input = vis_choice
        else:
            current_input = inject_input
            
        success, log_lines = run_script(script_cmd, inject_input=current_input)
        
        if success:
            metrics = extract_metrics(log_lines)
            results_dashboard[str(script_cmd)] = ("SUCCESS", metrics)
        else:
            results_dashboard[str(script_cmd)] = ("FAILED", ["Script crashed. Check pipeline_logs/."])
            
    # Print Final Dashboard
    print("\n\n" + "="*80)
    print("🏆 MASTER PIPELINE RESULTS DASHBOARD 🏆")
    print("="*80)
    
    for script, (status, metrics) in results_dashboard.items():
        status_icon = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⚠️"
        print(f"\n{status_icon} [ {script} ] -> {status}")
        for m in metrics:
            clean_m = " ".join(m.split())
            print(f"    ↳ {clean_m}")
            
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print("\n" + "="*80)
    print(f"Total Pipeline Execution Time: {int(hours)}h {int(mins)}m {secs:.2f}s")
    print("Logs saved in: ./pipeline_logs/")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
