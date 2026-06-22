import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, H, A):
        """
        H: (batch, N, in_features)
        A: (batch, N, N) - Adjacency matrix (assumed to be normalized or with self-loops)
        """
        H = self.dropout(H)
        # H_trans: (batch, N, out_features)
        H_trans = self.linear(H)
        # A * H_trans -> (batch, N, N) @ (batch, N, out_features) = (batch, N, out_features)
        H_next = torch.bmm(A, H_trans) + self.bias
        return H_next

class GNNEmbedding(nn.Module):
    def __init__(self, vocab_size=256, embedding_dim=64, hidden_dim=64, output_dim=128, dropout=0.5):
        """
        Lightweight Graph Convolutional Network (GCN) for DOM Structure representation.
        Takes node indices and dense adjacency matrices.
        """
        super(GNNEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2 Shallow Graph Convolution Layers to prevent over-smoothing
        self.gcn1 = GraphConvLayer(embedding_dim, hidden_dim, dropout=dropout)
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim, dropout=dropout)
        
        # Fully connected to produce final fixed-size embedding (128-D)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, nodes, adj):
        """
        nodes: (batch_size, max_nodes) - Node features (byte/char encoded indices)
        adj: (batch_size, max_nodes, max_nodes) - Dense adjacency matrix
        Returns: feature vector of size (batch_size, output_dim)
        """
        # Add self-loops to adjacency matrix for standard GCN message passing
        I = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0).expand_as(adj)
        A_tilde = adj + I
        
        # Normalize adjacency matrix A_hat = D^{-1/2} A_tilde D^{-1/2}
        D_tilde = torch.sum(A_tilde, dim=-1) # (batch, N)
        D_inv_sqrt = torch.pow(D_tilde, -0.5)
        D_inv_sqrt = torch.where(torch.isinf(D_inv_sqrt), torch.zeros_like(D_inv_sqrt), D_inv_sqrt)
        D_inv_sqrt_mat = torch.diag_embed(D_inv_sqrt) # (batch, N, N)
        
        A_hat = torch.bmm(torch.bmm(D_inv_sqrt_mat, A_tilde), D_inv_sqrt_mat)
        
        # (batch_size, max_nodes, emb_dim)
        H = self.embedding(nodes)
        
        # Layer 1
        H = self.gcn1(H, A_hat)
        H = F.relu(H)
        
        # Layer 2
        H = self.gcn2(H, A_hat)
        H = F.relu(H)
        
        # Global Max Pooling over the node dimension
        # H is (batch, max_nodes, hidden_dim) -> permute to (batch, hidden_dim, max_nodes)
        H_permuted = H.permute(0, 2, 1)
        pooled = F.max_pool1d(H_permuted, H_permuted.size(2)).squeeze(-1) # (batch, hidden_dim)
        
        # Final projection to fixed output dim
        embeddings = self.fc(pooled)
        
        return embeddings

if __name__ == "__main__":
    # Test the model and output shape
    model = GNNEmbedding(vocab_size=256, embedding_dim=64, hidden_dim=64, output_dim=128, dropout=0.5)
    batch_size = 4
    max_nodes = 100
    
    mock_nodes = torch.randint(0, 256, (batch_size, max_nodes))
    mock_adj = torch.randint(0, 2, (batch_size, max_nodes, max_nodes)).float()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    mock_nodes = mock_nodes.to(device)
    mock_adj = mock_adj.to(device)
    
    output = model(mock_nodes, mock_adj)
    print(f"Device: {device}")
    print(f"Nodes shape: {mock_nodes.shape}")
    print(f"Adj shape: {mock_adj.shape}")
    print(f"Output embedding shape: {output.shape}")
