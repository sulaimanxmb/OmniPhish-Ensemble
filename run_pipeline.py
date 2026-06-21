import os
import sys
import subprocess
import time
import re

SCRIPTS_TO_RUN = [
    ("trainer.py", "PROMPT_MODE"),  # Special flag to inject user's choice
    ("Check_for_overfitting.py", False),
    ("ablation_study.py", False),
    ("baselines/baseline_trainer.py", False),
    ("baselines/sota_trainer.py", False),
    ("baselines/sota2_trainer.py", False),
    ("generate_visualizations.py", False)
]

def run_script(script_name, inject_input=False):
    print(f"\n{'='*60}\n🚀 EXECUTING: {script_name}\n{'='*60}")
    
    log_dir = "pipeline_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    safe_name = script_name.replace("/", "_").replace("\\", "_")
    log_file = os.path.join(log_dir, f"{safe_name}.log")
    
    cmd = [sys.executable, script_name]
    
    try:
        if inject_input:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8', errors='ignore')
            process.stdin.write(f"{inject_input}\n")
            process.stdin.flush()
        else:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8', errors='ignore')
            
        full_log = []
        with open(log_file, "w", encoding="utf-8") as f:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
                full_log.append(line)
                
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[!] ERROR: {script_name} crashed with exit code {process.returncode}")
            return False, full_log
            
        return True, full_log
    except Exception as e:
        print(f"\n[!] Failed to execute {script_name}: {e}")
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

def main():
    start_time = time.time()
    
    print("\n" + "#"*80)
    print("OmniPhish - Master Research Pipeline Execution")
    print("#"*80)
    print("\nThis script will automatically execute the entire pipeline unattended.")
    print("All outputs will be mirrored to the 'pipeline_logs/' directory.\n")
    
    # Ask for Training Mode
    print("==================================================")
    print("🚀 SELECT TRAINING MODE FOR THE PIPELINE")
    print("==================================================")
    print("[1] FAST MODE (Global CodeBERT PEFT, Feature-Leak accepted, ~15 mins)")
    print("[2] SLOW MODE (Strict K-Fold CodeBERT Isolation, Zero-Leak, ~2 hours)")
    print("==================================================")
    mode_choice = input("Enter 1 or 2 [Default: 1]: ").strip()
    if mode_choice not in ['1', '2']:
        mode_choice = '1'
    
    results_dashboard = {}
    
    for script, inject_input in SCRIPTS_TO_RUN:
        if not os.path.exists(script):
            print(f"[!] Warning: {script} not found in this branch! Skipping.")
            results_dashboard[script] = ("SKIPPED", ["File not found"])
            continue
            
        # Resolve dynamic input
        current_input = mode_choice if inject_input == "PROMPT_MODE" else inject_input
            
        success, log_lines = run_script(script, inject_input=current_input)
        
        if success:
            metrics = extract_metrics(log_lines)
            results_dashboard[script] = ("SUCCESS", metrics)
        else:
            results_dashboard[script] = ("FAILED", ["Script crashed. Check pipeline_logs/."])
            
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
