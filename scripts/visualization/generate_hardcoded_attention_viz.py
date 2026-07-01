import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, RobertaModel
import transformers

transformers.logging.set_verbosity_error()

def generate_attention_viz():
    print("Loading CodeBERT...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    # CRITICAL: Added attn_implementation="eager" to support output_attentions=True
    model = RobertaModel.from_pretrained("microsoft/codebert-base", attn_implementation="eager")
    
    # Sample malicious JavaScript payload
    sample_code = "function verify() { var payload = eval(atob('YWxlcnQoJ3h4cycp')); return payload; }"
    
    inputs = tokenizer(sample_code, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    print("Running inference with output_attentions=True...")
    # Pass output_attentions=True to get the attention weights
    outputs = model(input_ids=input_ids, output_attentions=True)
    
    # attentions is a tuple of 12 layers. Each is shape: (batch_size, num_heads, seq_len, seq_len)
    attentions = outputs.attentions
    
    # We take the attention weights from the last layer (Layer 11)
    last_layer_attention = attentions[-1] 
    
    # Average the attention across all 12 heads in the last layer
    avg_attention = torch.mean(last_layer_attention, dim=1).squeeze(0) # Shape: (seq_len, seq_len)
    
    # We want to see how much the [CLS] token (index 0) attends to every other token
    cls_attention = avg_attention[0].detach().numpy()
    
    # Convert token IDs back to strings for the labels
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Create the visualization
    print("Generating visualization plot...")
    plt.figure(figsize=(12, 6))
    
    # We slice [1:-1] to remove the [CLS] and [SEP] tokens from the plot for cleaner viewing
    sns.barplot(x=tokens[1:-1], y=cls_attention[1:-1], color='crimson')
    
    plt.xticks(rotation=45, ha='right')
    plt.title("CodeBERT [CLS] Attention Weights (Last Layer, Avg Heads) on Obfuscated JS")
    plt.xlabel("Tokens")
    plt.ylabel("Attention Score")
    plt.tight_layout()
    
    # Save the plot
    # Updated path to not overwrite the dynamic visualization
    output_path = "visualizations/codebert_attention_hardcoded.png"
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Attention visualization saved to {output_path}")

if __name__ == "__main__":
    generate_attention_viz()
