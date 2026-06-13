import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMechanism(nn.Module):
    """
    Attention Mechanism based on the architectural diagram (Fig. 4).
    
    This module performs spatial attention by computing importance scores 
    for each location (H*W) in the feature maps, and then weighting the 
    original features accordingly.
    """
    def __init__(self, d_dim=256, h=16, w=16):
        """
        Args:
            d_dim: Number of input channels (D).
            h, w: Height and Width of the spatial feature map.
        """
        super(AttentionMechanism, self).__init__()
        self.l_dim = h * w
        
        # W: [D, 1] - Learnable weights to project D channels to a spatial score
        # b: [L, 1] - Learnable bias for each spatial location
        self.W = nn.Parameter(torch.randn(d_dim, 1))
        self.b = nn.Parameter(torch.zeros(self.l_dim, 1))

    def forward(self, x):
        """
        Input x: Feature maps [Batch, D, H, W]
        Output:  [Batch, D, H, W] focused features, [Batch, 1, L] attention weights
        """
        batch_size, D, H, W = x.shape
        L = H * W
        
        # 1. Reshape to [Batch, D, L] 
        # This matches the x ∈ R^{D x L} notation from the paper
        x_reshaped = x.view(batch_size, D, L)
        
        # 2. Score Computation: e_t = tanh(W^T * x + b)
        # Using matrix multiplication (W^T * x) produces a spatial score map
        # Result shape: [1, D] @ [Batch, D, L] -> [Batch, 1, L]
        scores = torch.matmul(self.W.t(), x_reshaped)
        
        # Add bias (broadcasted across batch) and apply activation
        scores = torch.tanh(scores + self.b.t())
        
        # 3. Attention Weights: a_t = softmax(e_t)
        # Normalize weights across the spatial dimension (L)
        attention_weights = F.softmax(scores, dim=2) # Shape: [Batch, 1, L]
        
        # 4. Element-wise weighting: Output = x * a_t
        # Broadcasting: [Batch, D, L] * [Batch, 1, L] applies the same 
        # weight to all channels at a specific spatial location.
        attended_features = x_reshaped * attention_weights
        
        # 5. Output: Restore original [Batch, D, H, W] dimensions
        return attended_features.view(batch_size, D, H, W), attention_weights

# --- Verification of Usage ---
# feature_maps = torch.randn(16, 256, 16, 16) # Batch size 16
# attention_module = AttentionMechanism(d_dim=256, h=16, w=16)
# focused_features, weights = attention_module(feature_maps)

# print(f"Output shape: {focused_features.shape}") 
# Expected: torch.Size([16, 256, 16, 16])
