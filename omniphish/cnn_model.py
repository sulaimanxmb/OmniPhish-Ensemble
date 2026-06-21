import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DEmbedding(nn.Module):
    def __init__(self, vocab_size=256, embedding_dim=64, num_filters=128, kernel_sizes=[3, 4, 5], output_dim=128):
        """
        Lightweight 1D-CNN for text/syntax representation.
        Takes character/byte level indices.
        """
        super(CNN1DEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Multiple convolution layers for different window sizes (n-grams)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=ks)
            for ks in kernel_sizes
        ])
        
        # Fully connected to produce final fixed-size embedding
        # output_dim represents the penultimate layer size
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_dim)
        
    def forward(self, x):
        """
        x: (batch_size, sequence_length) - byte/char encoded sequence
        Returns: feature vector of size (batch_size, output_dim)
        """
        # (batch_size, seq_len, emb_dim)
        embedded = self.embedding(x)
        
        # Conv1d expects (batch_size, in_channels, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolution + ReLU + Max Pooling over sequence
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # MaxPool1d over the whole sequence dimension: shape becomes (batch, num_filters, 1)
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conved]
        
        # Concatenate outputs from all kernel sizes
        # (batch_size, num_filters * len(kernel_sizes))
        cat = torch.cat(pooled, dim=1)
        
        # Pass through linear layer to get the final embeddings
        # This acts as the penultimate layer, discarding any task-specific classification head
        embeddings = self.fc(cat)
        
        return embeddings

def text_to_tensor(text, max_len=1024):
    """
    Helper function to convert text to byte-level tensor
    """
    bytes_list = list(text.encode('utf-8', errors='ignore'))
    # Truncate or pad
    if len(bytes_list) > max_len:
        bytes_list = bytes_list[:max_len]
    else:
        bytes_list += [0] * (max_len - len(bytes_list))
    # Return as batch of size 1
    return torch.tensor([bytes_list], dtype=torch.long)

if __name__ == "__main__":
    # Test the model and output shape
    model = CNN1DEmbedding(vocab_size=256, embedding_dim=32, num_filters=64, kernel_sizes=[3,4,5], output_dim=128)
    sample_text = "<html><body><h1>Hello World</h1></body></html>"
    input_tensor = text_to_tensor(sample_text, max_len=512)
    
    # Check if MPS is available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    output = model(input_tensor)
    print(f"Device: {device}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output embedding shape: {output.shape}")
