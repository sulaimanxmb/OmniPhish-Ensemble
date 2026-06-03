import os
import gc
import torch
import numpy as np

# Import phases
from html_parser import clean_html, extract_codebert_tags
from dom_to_graph import get_adjacency_and_features
from cnn_model import CNN1DEmbedding, text_to_tensor
from gnn_model import GNNDOMEmbedding
from transformer_model import CodeBERTEmbedding
from classifier import MetaClassifier

def process_html_file(filepath):
    print(f"\n--- Processing {filepath} ---")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_html = f.read()
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- PHASE 2: HTML Preprocessing & Parsing ---
    cleaned_html = clean_html(raw_html)
    codebert_text = extract_codebert_tags(cleaned_html)
    adj_matrix, node_features = get_adjacency_and_features(cleaned_html)

    # --- PHASE 3: 1D-CNN (Local Syntax) ---
    print("Loading CNN1DEmbedding...")
    cnn_model = CNN1DEmbedding().to(device)
    cnn_model.eval()
    
    cnn_input = text_to_tensor(cleaned_html, max_len=1024).to(device)
    
    with torch.no_grad():
        cnn_embedding = cnn_model(cnn_input)
        
    # Free memory
    print("Unloading CNN1DEmbedding to free memory...")
    del cnn_model
    del cnn_input
    gc.collect()
    
    # --- PHASE 4: GraphSAGE GNN (DOM Structure) ---
    print("Loading GNNDOMEmbedding...")
    gnn_model = GNNDOMEmbedding().to(device)
    gnn_model.eval()
    
    # Convert numpy to tensor
    gnn_x = torch.tensor(node_features, dtype=torch.float32).to(device)
    gnn_adj = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        gnn_embedding = gnn_model(gnn_x, gnn_adj)
        
    # Free memory
    print("Unloading GNNDOMEmbedding to free memory...")
    del gnn_model
    del gnn_x
    del gnn_adj
    gc.collect()

    # --- PHASE 5: CodeBERT (Logic & Obfuscation) ---
    print("Loading CodeBERTEmbedding...")
    codebert_model = CodeBERTEmbedding().to(device)
    codebert_model.eval()
    
    with torch.no_grad():
        codebert_embedding = codebert_model.compute_embedding(codebert_text)
        
    # Free memory
    print("Unloading CodeBERTEmbedding to free memory...")
    del codebert_model
    gc.collect()
    
    # Return features for the Meta-Classifier
    return cnn_embedding, gnn_embedding, codebert_embedding

def main():
    # Example orchestrated run on the Dataset
    dataset_dir = "dataset/raw_html"
    
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print(f"No HTML files found in {dataset_dir}. Run phish_scraper.py first.")
        return
        
    # Initialize the Meta Classifier
    meta_clf = MetaClassifier()
    features_list = []
    labels_list = []
    
    html_files = [f for f in os.listdir(dataset_dir) if f.endswith('.html')]
    
    for filename in html_files:
        filepath = os.path.join(dataset_dir, filename)
        
        try:
            cnn_feat, gnn_feat, cb_feat = process_html_file(filepath)
            
            # Combine into 1D Vector
            concat_vector = meta_clf.concatenate_features(cnn_feat, gnn_feat, cb_feat)
            features_list.append(concat_vector)
            
            # For demonstration, assign a dummy label (e.g. 0 for benign)
            labels_list.append(0)
            print(f"Successfully processed into feature vector of size {concat_vector.shape}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    if features_list:
        print("\n--- PHASE 6: Training Meta-Classifier ---")
        meta_clf.train(features_list, labels_list)
        print("Meta-Classifier trained successfully on extracted embeddings!")
        meta_clf.save("phishing_detector.pkl")
        print("Model saved to phishing_detector.pkl")

if __name__ == "__main__":
    main()
