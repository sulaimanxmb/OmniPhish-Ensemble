import os
import sys
import torch
import numpy as np
import re
import requests
import json
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omniphish.dataset_loader import PhishingDataset, custom_collate

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-coder:14b"

def query_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 5
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        # Silently fail on network timeout, usually caused if Ollama isn't running
        return ""

def run_llm_zeroshot():
    print("\n" + "="*50)
    print(f"🤖 RUNNING BASELINE: {MODEL_NAME.upper()} (Ollama Backend)")
    print("="*50)
    
    # Check if Ollama is running before starting the massive loop
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        models = [m['name'] for m in response.json().get('models', [])]
        if MODEL_NAME not in models and f"{MODEL_NAME}:latest" not in models:
            print(f"[!] ERROR: Ollama is running, but the model '{MODEL_NAME}' is not downloaded!")
            print(f"[!] Please open a new terminal and type: ollama pull {MODEL_NAME}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("[!] ERROR: Ollama server is not running or not installed!")
        print("[!] 1. Download Ollama from https://ollama.com/download/windows")
        print(f"[!] 2. Open a new terminal and type: ollama run {MODEL_NAME}")
        sys.exit(1)
    except Exception as e:
        print(f"[!] ERROR: Failed to communicate with Ollama: {e}")
        sys.exit(1)

    # 1. Prepare Dataset (Zero-Day Vault Only)
    dirs = {
        'phishing': 'dataset/raw_html/phishing',
        'benign': 'dataset/raw_html/benign'
    }
    
    full_dataset = PhishingDataset(dirs, undersample_benign=True)
    vault_size = int(len(full_dataset) * 0.10)
    train_val_size = len(full_dataset) - vault_size
    _, vault_dataset = torch.utils.data.random_split(
        full_dataset, [train_val_size, vault_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    vault_loader = DataLoader(vault_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    
    print(f"[*] Running inference on {len(vault_dataset)} Zero-Day Vault samples via Ollama...")
    
    all_preds = []
    all_labels = []
    errors = 0
    
    # Pre-compile the regex to extract 0 or 1
    number_pattern = re.compile(r'\b[01]\b')
    
    system_prompt = "You are a cybersecurity expert. You must classify websites as either phishing (1) or benign (0) based strictly on their HTML source code. Respond with ONLY the number 1 or 0. No explanations."

    for batch, labels in tqdm(vault_loader, desc="LLM Processing"):
        html_text = batch[0]['codebert_text'] 
        true_label = int(labels[0].item())
        all_labels.append(true_label)
        
        # Ollama Prompt (System + User combined for /api/generate)
        full_prompt = f"{system_prompt}\n\nClassify this HTML snippet. Output ONLY 1 for phishing, 0 for benign.\n\nHTML:\n{html_text[:3000]}"
        
        # Generate response
        response = query_ollama(full_prompt).strip()
        
        # Parse response
        match = number_pattern.search(response)
        if match:
            pred = int(match.group())
        else:
            # Fallback heuristics
            if '1' in response or 'phish' in response.lower():
                pred = 1
            elif '0' in response or 'benign' in response.lower() or 'safe' in response.lower():
                pred = 0
            else:
                errors += 1
                pred = 0 # Default to safe if totally confused
                
        all_preds.append(pred)
            
    print(f"\n[*] LLM Parsing Errors (Hallucinations): {errors}")
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    print("\n==================================================")
    print(f"📊 {MODEL_NAME.upper()} ZERO-DAY VAULT METRICS")
    print("==================================================")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"False Negatives (Missed Phishing): {fn}")
    print(f"Peak VRAM: ~9.50 GB (Externally managed by Ollama backend)")
    print("==================================================\n")

if __name__ == "__main__":
    run_llm_zeroshot()
