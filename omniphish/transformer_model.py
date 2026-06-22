import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel
import transformers

# We manually chunk sequences, so suppress the sequence length warnings
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CodeBERTEmbedding(nn.Module):
    def __init__(self, model_name="microsoft/codebert-base", use_lora=False):
        super(CodeBERTEmbedding, self).__init__()
        # CRITICAL FIX: AutoTokenizer automatically loads the Rust-based Fast Tokenizer
        # The pure Python RobertaTokenizer takes O(N^2) time on large HTML strings.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.codebert = RobertaModel.from_pretrained(model_name)
        
        # Freeze CodeBERT to save memory, as it acts as a static feature extractor
        for param in self.codebert.parameters():
            param.requires_grad = False
            
        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
                peft_config = LoraConfig(
                    task_type="FEATURE_EXTRACTION",
                    inference_mode=False,
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=["query", "value"]
                )
                self.codebert = get_peft_model(self.codebert, peft_config)
                for name, param in self.codebert.named_parameters():
                    if "lora" in name.lower():
                        param.requires_grad = True
                print("[CodeBERT] Initialized with LoRA (Low-Rank Adaptation).")
            except ImportError:
                print("\n[!] WARNING: 'peft' library not found. Run 'pip install peft'. CodeBERT will remain frozen without LoRA.\n")
                

    def compute_embedding(self, text, max_chunks=None):
        """
        Tokenizes text without truncation, applies 512-token chunks with 50-token overlap.
        Passes each chunk through CodeBERT independently and applies Global Max Pooling.
        Returns a single 768-D vector representing the most salient features.
        """
        chunk_size = 510 # Leave room for 2 special tokens: [CLS] and [SEP]
        overlap = 50
        stride = chunk_size - overlap
        
        # Determine chunk limit to prevent OOM
        if max_chunks is None:
            max_chunks = 4 if self.codebert.training else 12
            
        # VERY CRITICAL OPTIMIZATION:
        # BPE tokenizing a massive HTML file (millions of chars) takes forever on CPU.
        # Since we only use `max_chunks * chunk_size` tokens, we mathematically only need a fraction of the string.
        # Assuming ~3 chars per token, we slice the text generously to prevent massive CPU overhead.
        max_chars_needed = max_chunks * chunk_size * 5
        text = text[:max_chars_needed]
        
        # Fast Tokenization Hack to prevent O(N^2) slowdown on pure-Python tokenizer fallbacks
        chunk_size_chars = 2500
        input_ids_list = []
        attention_mask_list = []
        
        if len(text) == 0:
            text = " "
            
        for i in range(0, len(text), chunk_size_chars):
            sub_text = text[i:i+chunk_size_chars]
            inputs = self.tokenizer(
                sub_text, 
                return_tensors="pt", 
                truncation=False, 
                add_special_tokens=False 
            )
            input_ids_list.append(inputs["input_ids"][0])
            attention_mask_list.append(inputs["attention_mask"][0])
            
        input_ids = torch.cat(input_ids_list)
        attention_mask = torch.cat(attention_mask_list)
        
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    # Simulate a very long document
    sample_code = "<script>function login() { alert('test'); }</script>" * 200
    embedding = model.compute_embedding(sample_code)
    
    print(f"Device: {device}")
    print(f"Input text length (chars): {len(sample_code)}")
    print(f"Output Max-Pooled CodeBERT embedding shape: {embedding.shape}")
