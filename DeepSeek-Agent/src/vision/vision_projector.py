"""
Vision Language Projector

Maps vision tokens from the frozen DeepSeek encoder (2048-d) to the
language model embedding space (4096-d for Granite-7B).

This is the core trainable component in Stage 1.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VisionProjector(nn.Module):
    """
    Tiny 2-layer MLP projector: vision space (2048-d) → language space (4096-d).
    
    Architecture:
        Linear(2048 → 4096) → GELU → Linear(4096 → 4096) → LayerNorm + modality_embed
    
    Parameters: ~6M
    Training time: 2–4 hours on M4 Pro (1 epoch on 100k samples)
    """
    
    def __init__(
        self,
        in_dim: int = 2048,
        out_dim: int = 4096,
        hidden_dim: Optional[int] = None,
        use_layer_norm: bool = True,
        use_modality_embedding: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize the vision projector.
        
        Args:
            in_dim: Input dimension (DeepSeek token dim, typically 2048)
            out_dim: Output dimension (LLM hidden dim, typically 4096)
            hidden_dim: Hidden dimension (defaults to out_dim)
            use_layer_norm: Whether to apply LayerNorm to output
            use_modality_embedding: Whether to add learnable modality embedding
            dropout: Dropout rate in MLP layers
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = out_dim
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        
        # First linear layer with optional dropout
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        if dropout > 0:
            self.dropout1 = nn.Dropout(dropout)
        else:
            self.dropout1 = None
        
        # Second linear layer
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        if dropout > 0:
            self.dropout2 = nn.Dropout(dropout)
        else:
            self.dropout2 = None
        
        # Optional LayerNorm
        if use_layer_norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None
        
        # Learnable modality embedding: signals the model that input is vision
        if use_modality_embedding:
            self.modality_embed = nn.Parameter(
                torch.randn(1, 1, out_dim) * (1.0 / (out_dim ** 0.5))
            )
        else:
            self.modality_embed = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate scales."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """
        Project vision tokens from 2048-d to 4096-d space.
        
        Args:
            vision_tokens: Tensor of shape (B, N, 2048)
        
        Returns:
            Projected tokens of shape (B, N, 4096)
        """
        # First layer
        x = self.lin1(vision_tokens)  # (B, N, 2048) → (B, N, hidden_dim)
        x = self.gelu(x)
        if self.dropout1 is not None:
            x = self.dropout1(x)
        
        # Second layer
        x = self.lin2(x)  # (B, N, hidden_dim) → (B, N, 4096)
        if self.dropout2 is not None:
            x = self.dropout2(x)
        
        # Optional LayerNorm
        if self.norm is not None:
            x = self.norm(x)
        
        # Add modality embedding
        if self.modality_embed is not None:
            x = x + self.modality_embed
        
        return x
    
    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


class VisionProjectorWithAttention(nn.Module):
    """
    Alternative projector using cross-attention for more sophisticated alignment.
    Useful if you want to learn non-linear mappings per-token.
    """
    
    def __init__(
        self,
        in_dim: int = 2048,
        out_dim: int = 4096,
        num_heads: int = 8,
        num_layers: int = 1,
    ):
        """
        Initialize attention-based vision projector.
        
        Args:
            in_dim: Input dimension (2048)
            out_dim: Output dimension (4096)
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Initial projection to match attention dimension
        self.embed = nn.Linear(in_dim, out_dim)
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=out_dim,
                nhead=num_heads,
                dim_feedforward=out_dim * 4,
                batch_first=True,
                dropout=0.1,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """
        Project vision tokens using cross-attention.
        
        Args:
            vision_tokens: Tensor of shape (B, N, 2048)
        
        Returns:
            Projected tokens of shape (B, N, 4096)
        """
        x = self.embed(vision_tokens)  # (B, N, 2048) → (B, N, 4096)
        
        for layer in self.layers:
            x = layer(x)  # Apply transformer block
        
        x = self.norm(x)
        return x


if __name__ == "__main__":
    # Example usage
    projector = VisionProjector(in_dim=2048, out_dim=4096)
    
    # Dummy vision tokens from DeepSeek encoder
    vision_tokens = torch.randn(4, 576, 2048)
    
    # Project to LLM space
    projected = projector(vision_tokens)
    
    print(f"Input shape: {vision_tokens.shape}")
    print(f"Output shape: {projected.shape}")
    print(f"Number of parameters: {projector.get_num_params():,}")
