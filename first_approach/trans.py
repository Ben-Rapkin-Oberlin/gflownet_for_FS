import torch
import torch.nn as nn
import math

class FeatureSelectionTransformer(nn.Module):
    """
    Transformer-based architecture for feature selection GFlowNet.
    """
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Embedding for features (both selected and unselected)
        self.feature_embedding = nn.Embedding(n_features, d_model)
        
        # Position encoding for selected/unselected status
        self.selection_embedding = nn.Embedding(2, d_model)  # 0=unselected, 1=selected
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection (can be used for different heads: PF, PB, logF)
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, states: torch.Tensor):
        """
        Args:
            states: Binary tensor of shape (batch_size, n_features) 
                   indicating selected features
        """
        batch_size, n_features = states.shape
        
        # Create feature indices
        feature_indices = torch.arange(n_features, device=states.device)
        feature_indices = feature_indices.expand(batch_size, -1)
        
        # Get embeddings
        feature_emb = self.feature_embedding(feature_indices)  # Shape: (batch, n_features, d_model)
        selection_emb = self.selection_embedding(states.long())  # Shape: (batch, n_features, d_model)
        
        # Combine embeddings
        x = feature_emb + selection_emb
        
        # Apply transformer
        x = self.transformer(x)  # Shape: (batch, n_features, d_model)
        
        # Global pooling and projection
        x = torch.mean(x, dim=1)  # Shape: (batch, d_model)
        x = self.output_projection(x)  # Shape: (batch, d_model)
        
        return x

class TransformerPolicyHead(nn.Module):
    """Policy head with proper numerical stability."""
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim)
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x):
        # Apply network
        logits = self.head(x)
        # Don't apply softmax here - we'll handle it in the policy computation
        return logits

class TransformerLogFHead(nn.Module):
    """
    LogF head for the transformer architecture.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x):
        return self.head(x)