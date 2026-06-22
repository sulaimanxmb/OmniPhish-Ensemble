import os
import sys
import subprocess
import time
import re

# ANSI Color Codes for Aesthetics
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
MAGENTA = "\033[1;35m"
RESET = "\033[0m"
BOLD = "\033[1m"

SCRIPTS_TO_RUN = [
    ("scripts/training/trainer.py", "PROMPT_MODE"),  # Special flag to inject user's choice
    ("scripts/training/Check_for_overfitting.py", False),
    ("scripts/training/ablation_study.py", False),
    ("baselines/baseline_trainer.py", False),
    ("baselines/sota_trainer.py", False),
    ("baselines/sota2_trainer.py", False),
    ("scripts/visualization/generate_visualizations.py", False)
]

def run_script(script_name, inject_input=False):
    print(f"\n{CYAN}{'━'*80}{RESET}")
    print(f"{BOLD}{MAGENTA}🚀 EXECUTING:{RESET} {CYAN}{script_name}{RESET}")
    print(f"{CYAN}{'━'*80}{RESET}")
    
    log_dir = "pipeline_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    safe_name = script_name.replace("/", "_").replace("\\", "_")
    log_file = os.path.join(log_dir, f"{safe_name}.log")
    
    cmd = [sys.executable, script_name]
    
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TQDM_FORCE_TTY"] = "True"  # Force tqdm to use \r animations
        env["FORCE_COLOR"] = "1"
        
        if inject_input:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, env=env)
            process.stdin.write(f"{inject_input}\n".encode('utf-8'))
            process.stdin.flush()
        else:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, env=env)
            
        full_log = []
        current_line = bytearray()
        
        with open(log_file, "w", encoding="utf-8") as f:
            while True:
                byte = process.stdout.read(1)
                if not byte:
                    break
                    
                # Write to terminal instantly
                sys.stdout.buffer.write(byte)
                sys.stdout.buffer.flush()
                
                # Handle log file buffering
                if byte == b'\n':
                    line_str = current_line.decode('utf-8', errors='ignore')
                    # Windows uses \r\n, and tqdm uses \r to overwrite lines.
                    # By stripping trailing \r and splitting by \r, we perfectly isolate
                    # the final state of the progress bar, or the normal print statement!
                    final_str = line_str.rstrip('\r').split('\r')[-1]
                    f.write(final_str + '\n')
                    full_log.append(final_str + '\n')
                    current_line.clear()
                else:
                    current_line.extend(byte)
                
        process.wait()
        
        if process.returncode != 0:
            print(f"\n{RED}[!] ERROR: {script_name} crashed with exit code {process.returncode}{RESET}")
            return False, full_log
            
        return True, full_log
    except Exception as e:
        print(f"\n{RED}[!] Failed to execute {script_name}: {e}{RESET}")
        return False, []

def extract_metrics(log_lines):
    """Scans the bottom of the log to extract detailed evaluation metrics."""
    metrics = []
    # Keywords to look for in the final output
    keywords = ['accuracy', 'precision', 'recall', 'f1-score', 'f1 score', 'roc-auc', 'mcc', 'fpr', 'auc']
    
    for line in log_lines[-150:]:
        line_lower = line.lower()
        # Look for Average metrics from K-Fold or direct metrics
        if any(f"{kw}:" in line_lower or f"{kw} :" in line_lower for kw in keywords) or "average " in line_lower:
            # Clean up the line for the dashboard
            clean_line = line.strip().replace("\r", "")
            if clean_line and clean_line not in metrics:
                # To prevent grabbing huge scikit-learn classification reports, we skip lines that don't have a colon
                if ":" in clean_line:
                    metrics.append(clean_line)
                    
    # Only keep the last 15 matching metrics to avoid flooding the dashboard if a script prints them multiple times
    return metrics[-15:] if metrics else ["Completed successfully (No generic metrics extracted). Check logs."]

def check_dataset_integrity():
    """Validates that the exact expected number of raw HTML files exist to prevent corrupted runs."""
    benign_dir = os.path.join("Dataset", "raw_html", "benign")
    phish_dir = os.path.join("Dataset", "raw_html", "phishing")
    
    if not os.path.exists(benign_dir) or not os.path.exists(phish_dir):
        print(f"\n{RED}[!] FATAL ERROR: Dataset directories not found! Please extract Dataset.zip into the root folder.{RESET}")
        sys.exit(1)
        
    benign_count = len([f for f in os.listdir(benign_dir) if f.endswith(".html")])
    phish_count = len([f for f in os.listdir(phish_dir) if f.endswith(".html")])
    
    EXPECTED_BENIGN = 2250
    EXPECTED_PHISH = 5740
    
    if benign_count != EXPECTED_BENIGN or phish_count != EXPECTED_PHISH:
        print(f"\n{RED}{'!'*80}")
        print("🚨 DATASET CORRUPTION DETECTED 🚨")
        print(f"Expected Benign: {EXPECTED_BENIGN} | Found: {benign_count}")
        print(f"Expected Phishing: {EXPECTED_PHISH} | Found: {phish_count}")
        print("Please re-extract the OmniPhish Kaggle dataset cleanly to prevent corrupted research metrics.")
        print(f"{'!'*80}{RESET}\n")
        sys.exit(1)
        
    print(f"{GREEN}[*] Dataset Integrity Verified: {benign_count} Benign, {phish_count} Phishing.{RESET}")

def print_banner():
    banner = f"""
{CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  {BOLD}🛡️  OMNIPHISH ENSEMBLE  🛡️{RESET}{CYAN}                                                  ║
║  {YELLOW}Master Research Pipeline Execution{RESET}{CYAN}                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)
    print(f"{GREEN}Automated pipeline initialized. All outputs will be mirrored to 'pipeline_logs/'.{RESET}\n")

def main():
    print_banner()
    check_dataset_integrity()
    start_time = time.time()
    
    global SCRIPTS_TO_RUN
    
    print(f"\n{CYAN}╭──────────────────────────────────────────────────╮{RESET}")
    print(f"{CYAN}│{RESET} {BOLD}⚙️  PIPELINE CONFIGURATION{RESET}                         {CYAN}│{RESET}")
    print(f"{CYAN}╰──────────────────────────────────────────────────╯{RESET}")
    print(f" {GREEN}[1]{RESET} RUN trainer.py (Execute full training from scratch)")
    print(f" {YELLOW}[2]{RESET} SKIP trainer.py (Only evaluations and baselines using existing weights)")
    print(f"{CYAN}────────────────────────────────────────────────────{RESET}")
    skip_choice = input(f"{BOLD}Enter 1 or 2 [Default: 1]:{RESET} ").strip()
    
    mode_choice = '1'
    if skip_choice == '2':
        SCRIPTS_TO_RUN = [s for s in SCRIPTS_TO_RUN if "trainer.py" not in s[0]]
    else:
        print(f"\n{CYAN}╭──────────────────────────────────────────────────╮{RESET}")
        print(f"{CYAN}│{RESET} {BOLD}⚡ SELECT TRAINING MODE{RESET}                           {CYAN}│{RESET}")
        print(f"{CYAN}╰──────────────────────────────────────────────────╯{RESET}")
        print(f" {GREEN}[1]{RESET} FAST MODE (Global CodeBERT PEFT, Feature-Leak accepted, ~15 mins)")
        print(f" {YELLOW}[2]{RESET} SLOW MODE (Strict K-Fold CodeBERT Isolation, Zero-Leak, ~2 hours)")
        print(f"{CYAN}────────────────────────────────────────────────────{RESET}")
        mode_choice = input(f"{BOLD}Enter 1 or 2 [Default: 1]:{RESET} ").strip()
        if mode_choice not in ['1', '2']:
            mode_choice = '1'
    
    results_dashboard = {}
    
    for script, inject_input in SCRIPTS_TO_RUN:
        if not os.path.exists(script):
            print(f"{YELLOW}[!] Warning: {script} not found in this branch! Skipping.{RESET}")
            results_dashboard[script] = ("SKIPPED", ["File not found"])
            continue
            
        current_input = mode_choice if inject_input == "PROMPT_MODE" else inject_input
        success, log_lines = run_script(script, inject_input=current_input)
        
        if success:
            metrics = extract_metrics(log_lines)
            results_dashboard[script] = ("SUCCESS", metrics)
        else:
            results_dashboard[script] = ("FAILED", ["Script crashed. Check pipeline_logs/."])
            
    # Print Final Dashboard
    print(f"\n\n{MAGENTA}╔══════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{MAGENTA}║{RESET} {BOLD}🏆 MASTER PIPELINE RESULTS DASHBOARD 🏆{RESET}                                      {MAGENTA}║{RESET}")
    print(f"{MAGENTA}╚══════════════════════════════════════════════════════════════════════════════╝{RESET}")
    
    for script, (status, metrics) in results_dashboard.items():
        if status == "SUCCESS":
            status_display = f"{GREEN}✅ SUCCESS{RESET}"
        elif status == "FAILED":
            status_display = f"{RED}❌ FAILED{RESET}"
        else:
            status_display = f"{YELLOW}⚠️ SKIPPED{RESET}"
            
        print(f"\n {BOLD}[ {CYAN}{script}{RESET} ]{RESET} ──────── {status_display}")
        for m in metrics:
            clean_m = " ".join(m.split())
            print(f"    ↳ {clean_m}")
            
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    
    print(f"\n{CYAN}{'━'*80}{RESET}")
    print(f"{BOLD}⏳ Total Pipeline Execution Time: {int(hours)}h {int(mins)}m {secs:.2f}s{RESET}")
    print(f"{GREEN}📂 Logs perfectly saved in: ./pipeline_logs/{RESET}")
    print(f"{CYAN}{'━'*80}{RESET}\n")

if __name__ == "__main__":
    main()
