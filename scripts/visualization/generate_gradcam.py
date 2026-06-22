import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import torch
import torch.nn.functional as F
import numpy as np

from omniphish.cnn_model import CNN1DEmbedding, text_to_tensor

# Global variables to store intermediate math
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def generate_gradcam():
    print("="*60)
    print("🔬 1D GRAD-CAM XAI ANALYZER (CNN Feature Extraction)")
    print("="*60)
    
    # 1. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    target_feature_idx = 106  # From XGBoost Feature Importance
    
    print("[*] Loading PyTorch CNN Model...")
    model = CNN1DEmbedding(embedding_dim=64, num_filters=128, output_dim=128).to(device)
    model.load_state_dict(torch.load("weights/cnn_trained.pt", map_location=device))
    model.eval()
    
    # 2. Inject Hooks into the Convolutional Layer (Kernel Size 3)
    print("[*] Injecting Mathematical Hooks into Convolution Layer 0...")
    model.convs[0].register_forward_hook(forward_hook)
    model.convs[0].register_full_backward_hook(backward_hook)
    
    # 3. Choose Phishing or Benign
    print("\nSelect the type of site to analyze:")
    print("  [1] Zero-Day Phishing Site (Default)")
    print("  [2] Legitimate Benign Site")
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '2':
        target_dir = "dataset/raw_html/benign"
        site_type = "Benign"
    else:
        target_dir = "dataset/raw_html/phishing"
        site_type = "Phishing"
        
    print(f"\n[*] Selecting a random {site_type} Sample...")
    if not os.path.exists(target_dir) or len(os.listdir(target_dir)) == 0:
        print(f"[!] Error: Cannot find samples in {target_dir}")
        return
        
    all_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    import random
    random_file = random.choice(all_files)
    sample_filename = os.path.basename(random_file)
    sample_path = random_file
    print(f"[*] Selected: {sample_filename} ({os.path.getsize(sample_path) / 1024:.2f} KB)")
    
    with open(sample_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_text = f.read()
        
    # Truncate to 1024 bytes (the CNN input limit)
    raw_text = raw_text[:1024]
    input_tensor = text_to_tensor(raw_text, max_len=1024).to(device)
    
    # 4. Forward Pass
    print(f"[*] Running Forward Pass to isolate CNN_Feat_{target_feature_idx}...")
    output = model(input_tensor)
    
    # 5. Backward Pass (Treat the specific feature as the target score!)
    print(f"[*] Backpropagating gradients from Feature {target_feature_idx} into the Convolutions...")
    model.zero_grad()
    target_score = output[0, target_feature_idx]
    target_score.backward(retain_graph=True)
    
    # 6. Calculate Grad-CAM Heat
    print("[*] Calculating 1D Heat Map...")
    global activations, gradients
    
    # Average the gradients across the channels to get the feature weights
    weights = torch.mean(gradients, dim=2, keepdim=True)
    
    # Multiply weights by activations and sum across channels
    cam = torch.sum(weights * activations, dim=1).squeeze()
    
    # Apply ReLU to only keep positive influence, then normalize between 0 and 1
    cam = F.relu(cam)
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > 0:
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    
    cam = cam.cpu().detach().numpy()
    
    # The CNN kernel size is 3, so the output is 1022. We pad 1 zero to the left and right to match 1024.
    cam_padded = np.pad(cam, (1, 1), mode='constant', constant_values=0)
    
    # 7. Generate Visual HTML Report
    print("[*] Generating Visual HTML Report...")
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "gradcam_report.html")
    
    html_content = f"""
    <html>
    <head>
        <title>OmniPhish 1D Grad-CAM</title>
        <style>
            body {{ font-family: monospace; background-color: #1e1e1e; color: #d4d4d4; padding: 20px; }}
            .hot-high {{ background-color: #ff0000; color: white; font-weight: bold; }}
            .hot-med {{ background-color: #ff6666; color: white; font-weight: bold; }}
            .hot-low {{ background-color: #ffb3b3; color: black; }}
            .cold {{ background-color: transparent; }}
            .header {{ background-color: #333; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🔬 1D Grad-CAM Explainability Report</h2>
            <p><strong>Target Feature:</strong> CNN_Feat_{target_feature_idx}</p>
            <p><strong>Analyzed File:</strong> {sample_filename} ({site_type})</p>
            <p><em>Red highlighting indicates the exact structural HTML sequence that triggers the CNN's Phishing detection.</em></p>
        </div>
        <div style="white-space: pre-wrap; word-wrap: break-word;">
    """
    
    for i, char in enumerate(raw_text):
        heat = cam_padded[i] if i < len(cam_padded) else 0
        escaped_char = char.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        if heat > 0.7:
            html_content += f'<span class="hot-high">{escaped_char}</span>'
        elif heat > 0.4:
            html_content += f'<span class="hot-med">{escaped_char}</span>'
        elif heat > 0.1:
            html_content += f'<span class="hot-low">{escaped_char}</span>'
        else:
            html_content += f'<span class="cold">{escaped_char}</span>'
            
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\n[+] SUCCESS! Open '{report_path}' in your browser to view the XAI results!")
    print("="*60)

if __name__ == "__main__":
    generate_gradcam()
