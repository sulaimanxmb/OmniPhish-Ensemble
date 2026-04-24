import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from peft import LoraConfig, get_peft_model

class CodeBERTEmbedding(nn.Module):
    def __init__(self, model_name="microsoft/codebert-base", use_lora=True):
        super(CodeBERTEmbedding, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        base_model = RobertaModel.from_pretrained(model_name)
        
        self.use_lora = use_lora
        if use_lora:
            # Configure LoRA to adapt the query and value attention matrices
            config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.05,
                bias="none",
                modules_to_save=["pooler"] # Save pooler if used, but we use CLS token directly
            )
            self.codebert = get_peft_model(base_model, config)
            # PEFT automatically sets requires_grad=True only for LoRA parameters
            self.codebert.print_trainable_parameters()
        else:
            self.codebert = base_model
            for param in self.codebert.parameters():
                param.requires_grad = False
                
    def train_lora_mode(self):
        """Sets the model to training mode for Phase 1.5."""
        if self.use_lora:
            self.codebert.train()
            
    def eval_lora_and_freeze(self):
        """Sets to eval mode and freezes everything for Phase 2 extraction."""
        self.codebert.eval()
        for param in self.codebert.parameters():
            param.requires_grad = False
            
    def compute_embedding(self, text, max_chunks=None):
        """
        Tokenizes text without truncation, applies 512-token chunks with 50-token overlap.
        Passes each chunk through CodeBERT independently and applies Global Max Pooling.
        Returns a single 768-D vector representing the most salient features.
        """
        # Suppress the max length warning by temporarily changing logging level, 
        # or we just let it warn once. It's harmless since we chunk manually.
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=False, 
            add_special_tokens=False 
        )
        
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        
        chunk_size = 510 # Leave room for 2 special tokens: [CLS] and [SEP]
        overlap = 50
        stride = chunk_size - overlap
        
        device = next(self.codebert.parameters()).device
        embeddings = []
        
        # If text is empty
        if len(input_ids) == 0:
            empty_input = self.tokenizer("", return_tensors="pt", padding="max_length", max_length=512)
            c_ids = empty_input["input_ids"].to(device)
            c_mask = empty_input["attention_mask"].to(device)
            with torch.set_grad_enabled(self.codebert.training):
                outputs = self.codebert(input_ids=c_ids, attention_mask=c_mask)
                return outputs.last_hidden_state[:, 0, :].squeeze(0)
        
        # Determine chunk limit to prevent OOM (Out of Memory)
        # If training (gradients enabled), limit chunks strictly.
        if max_chunks is None:
            max_chunks = 4 if self.codebert.training else 12
            
        chunk_ids_list = []
        chunk_masks_list = []
        
        chunk_count = 0
        for i in range(0, len(input_ids), stride):
            if chunk_count >= max_chunks:
                break
                
            chunk_ids = input_ids[i:i + chunk_size]
            chunk_mask = attention_mask[i:i + chunk_size]
            
            # Add [CLS] and [SEP]
            cls_token = torch.tensor([self.tokenizer.cls_token_id])
            sep_token = torch.tensor([self.tokenizer.sep_token_id])
            mask_ones = torch.tensor([1])
            
            chunk_ids = torch.cat([cls_token, chunk_ids, sep_token])
            chunk_mask = torch.cat([mask_ones, chunk_mask, mask_ones])
            
            # Pad to 512
            if len(chunk_ids) < 512:
                pad_len = 512 - len(chunk_ids)
                pad_ids = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                pad_mask = torch.zeros((pad_len,), dtype=torch.long)
                
                chunk_ids = torch.cat([chunk_ids, pad_ids])
                chunk_mask = torch.cat([chunk_mask, pad_mask])
                
            chunk_ids_list.append(chunk_ids)
            chunk_masks_list.append(chunk_mask)
            
            chunk_count += 1
            if i + chunk_size >= len(input_ids):
                break
                
        # Batch execute all chunks for this document simultaneously (10x faster than sequential loop)
        batch_c_ids = torch.stack(chunk_ids_list).to(device)
        batch_c_masks = torch.stack(chunk_masks_list).to(device)
        
        with torch.set_grad_enabled(self.codebert.training):
            outputs = self.codebert(input_ids=batch_c_ids, attention_mask=batch_c_masks)
            all_embs = outputs.last_hidden_state[:, 0, :] # Shape: (num_chunks, 768)
            
        max_pooled_emb, _ = torch.max(all_embs, dim=0) # Shape: (768,)
        
        return max_pooled_emb

if __name__ == "__main__":
    model = CodeBERTEmbedding(use_lora=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    # Simulate a very long document
    sample_code = "<script>function login() { alert('test'); }</script>" * 200
    embedding = model.compute_embedding(sample_code)
    
    print(f"Device: {device}")
    print(f"Input text length (chars): {len(sample_code)}")
    print(f"Output Max-Pooled CodeBERT embedding shape: {embedding.shape}")
