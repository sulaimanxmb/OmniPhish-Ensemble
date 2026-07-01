import sys
import os
import argparse
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, RobertaModel
import transformers

# Ensure the root directory is on the path so we can import omniphish
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from omniphish.dataset_loader import PhishingDataset

OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_attention_viz(sample_type):
    print(f"\n[*] Generating CodeBERT Semantic Attention Visualization for a random {sample_type.upper()} sample...")
    transformers.logging.set_verbosity_error()
    
    print("    [+] Loading CodeBERT Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base", attn_implementation="eager")
    
    print(f"    [+] Loading Dataset...")
    dirs = {'phishing': 'dataset/raw_html/phishing', 'benign': 'dataset/raw_html/benign'}
    dataset = PhishingDataset(dirs, undersample_benign=False)
    
    label_target = 1 if sample_type == "phishing" else 0
    indices = [i for i, label in enumerate(dataset.labels) if label == label_target]
    
    if not indices:
        print(f"    [!] No {sample_type} samples found in the dataset.")
        return
        
    success = False
    js_text = ""
    filename = ""
    
    # Try multiple times to find a sample with enough content
    for attempt in range(20):
        rand_idx = random.choice(indices)
        sample = dataset[rand_idx]
        js_text = sample['codebert_text']
        
        if len(js_text.strip()) > 50:
            filepath = dataset.samples[rand_idx]
            # Strip the \\?\ prefix if it exists on Windows
            if filepath.startswith('\\\\?\\'):
                filepath = filepath[4:]
            filename = os.path.basename(filepath)
            success = True
            break
            
    if not success:
        print(f"    [!] Could not find a {sample_type} sample with sufficient content.")
        return
        
    print(f"    [+] Selected {sample_type.capitalize()} Sample:")
    print(f"        -> File: {filename}")
    print(f"        -> Content Length: {len(js_text)} characters")
    
    snippet = js_text[:300]
    inputs = tokenizer(snippet, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    if input_ids.shape[1] > 60:
        input_ids = input_ids[:, :60]
        
    print("    [+] Extracting Attention Weights (Layer 11, Avg Heads)...")
    outputs = model(input_ids=input_ids, output_attentions=True)
    attentions = outputs.attentions
    
    last_layer_attention = attentions[-1] 
    avg_attention = torch.mean(last_layer_attention, dim=1).squeeze(0)
    cls_attention = avg_attention[0].detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    plt.figure(figsize=(14, 6))
    sns.barplot(x=tokens[1:-1], y=cls_attention[1:-1], color='crimson' if sample_type == 'phishing' else 'royalblue')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    plt.title(f"CodeBERT Attention Weights [{sample_type.capitalize()} Sample: {filename}]", fontweight='bold', pad=15)
    plt.xlabel("Extracted HTML/JS Tokens", fontweight='bold')
    plt.ylabel("Attention Score (CLS to Token)", fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_file = os.path.join(OUTPUT_DIR, f"codebert_attention_{sample_type}_{filename}.png")
    plt.savefig(out_file, dpi=300)
    print(f"    [+] Saved visualization to '{out_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CodeBERT Attention Visualization")
    parser.add_argument("--type", type=str, choices=["phishing", "benign"], default="phishing", 
                        help="Which class of sample to randomly visualize")
    args = parser.parse_args()
    
    print("="*60)
    print("OMNIPHISH ATTENTION VISUALIZER")
    print("="*60)
    generate_attention_viz(args.type)
    print("="*60)
