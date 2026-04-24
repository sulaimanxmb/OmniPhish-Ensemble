import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

class CodeBERTEmbedding(nn.Module):
    def __init__(self, model_name="microsoft/codebert-base"):
        super(CodeBERTEmbedding, self).__init__()
        # Load the base model without the classification head
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.codebert = RobertaModel.from_pretrained(model_name)
        
        # We freeze CodeBERT to save memory, as we only use it for feature extraction
        for param in self.codebert.parameters():
            param.requires_grad = False
            
    def compute_embedding(self, text):
        """
        Takes raw string (extracted tags), tokenizes with max 512 tokens,
        and returns the [CLS] embedding of size 768.
        """
        # Strict truncation and padding
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            padding="max_length", 
            truncation=True
        )
        
        # Move tensors to the same device as the model
        device = next(self.codebert.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Output [CLS] embedding
        with torch.no_grad():
            outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
            # The pooler_output or simply taking the first token representation
            # outputs.last_hidden_state shape: (batch_size, sequence_length, hidden_size)
            # [CLS] token is the first token
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
        return cls_embedding
        
if __name__ == "__main__":
    model = CodeBERTEmbedding()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    sample_code = "<script>function login() { alert('test'); }</script>"
    embedding = model.compute_embedding(sample_code)
    
    print(f"Device: {device}")
    print(f"Input text length (chars): {len(sample_code)}")
    print(f"Output CodeBERT embedding shape: {embedding.shape}")
