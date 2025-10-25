"""
Vision Token Compressor

Reduces the number of vision tokens (e.g., 576 → 32) for efficient
long-context LLM inference while preserving key visual information.

Uses learned attention pooling (a form of Perceiver-style resampling).
"""

import logging
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VisionTokenCompressor(nn.Module):
    """
    Compresses vision tokens via learned attention pooling.
    
    Maps (B, N, 4096) → (B, K, 4096), where K is configurable (default 32).
    
    This reduces context pressure for long-context LLMs while preserving
    the most informative visual features.
    """
    
    def __init__(
        self,
        num_output_tokens: int = 32,
        hidden_dim: int = 4096,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize the vision token compressor.
        
        Args:
            num_output_tokens: Number of compressed tokens (default 32)
            hidden_dim: Dimension of the tokens (should match projector output)
            num_heads: Number of attention heads in the pooling mechanism
            dropout: Dropout rate for attention
        """
        super().__init__()
        
        self.num_output_tokens = num_output_tokens
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Learned query vectors (tokens to pool into)
        self.queries = nn.Parameter(
            torch.randn(1, num_output_tokens, hidden_dim) * (1.0 / (hidden_dim ** 0.5))
        )
        
        # Multi-head attention: queries attend to input tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        
        # Post-attention normalization and projection
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compress vision tokens.
        
        Args:
            vision_tokens: Tensor of shape (B, N, 4096)
        
        Returns:
            Compressed tokens of shape (B, K, 4096), where K = num_output_tokens
        """
        B = vision_tokens.shape[0]
        
        # Expand query vectors for batch
        queries = self.queries.expand(B, -1, -1)  # (B, K, 4096)
        
        # Multi-head attention: queries attend to input tokens
        # This pools information from all input tokens into the queries
        attn_output, _ = self.attn(
            queries,  # Query
            vision_tokens,  # Key & Value
            vision_tokens,
        )  # → (B, K, 4096)
        
        # Residual connection + norm
        x = self.norm(attn_output + queries)
        
        # MLP projection
        mlp_output = self.mlp(x)
        
        # Residual connection + norm
        output = self.norm2(mlp_output + x)
        
        return output  # (B, K, 4096)
    
    def get_compression_ratio(self, num_input_tokens: int) -> float:
        """Return the compression ratio."""
        return num_input_tokens / self.num_output_tokens


class VisionTokenCompressorSimple(nn.Module):
    """
    Simple strided pooling compressor.
    
    Faster but less learnable than attention-based compression.
    """
    
    def __init__(
        self,
        stride: int = 16,
        use_mean: bool = True,
    ):
        """
        Initialize simple compressor.
        
        Args:
            stride: Pooling stride (e.g., stride=16 reduces 576 tokens to ~36)
            use_mean: If True, average pool; if False, take max
        """
        super().__init__()
        self.stride = stride
        self.use_mean = use_mean
    
    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compress by strided pooling.
        
        Args:
            vision_tokens: Tensor of shape (B, N, D)
        
        Returns:
            Compressed tokens of shape (B, N // stride, D)
        """
        # Simple strided selection
        compressed = vision_tokens[:, ::self.stride, :]
        
        if self.use_mean and self.stride > 1:
            # More sophisticated: average over stride windows
            B, N, D = vision_tokens.shape
            
            # Pad to multiple of stride
            pad_len = (self.stride - (N % self.stride)) % self.stride
            padded = torch.nn.functional.pad(vision_tokens, (0, 0, 0, pad_len))
            
            # Reshape and mean
            B_p, N_p, D_p = padded.shape
            reshaped = padded.view(B_p, -1, self.stride, D_p)  # (B, N//stride, stride, D)
            compressed = reshaped.mean(dim=2)  # (B, N//stride, D)
        
        return compressed


class VisionTokenPerceiver(nn.Module):
    """
    Perceiver-style resampler for vision tokens.
    
    More sophisticated compression using a small transformer to
    aggregate information from many input tokens into fewer output tokens.
    """
    
    def __init__(
        self,
        num_output_tokens: int = 32,
        input_dim: int = 4096,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """
        Initialize Perceiver-style resampler.
        
        Args:
            num_output_tokens: Number of output tokens
            input_dim: Dimension of input tokens
            depth: Number of resampler blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to input dim
            dropout: Dropout rate
        """
        super().__init__()
        
        # Learnable latent tokens
        self.latents = nn.Parameter(
            torch.randn(1, num_output_tokens, input_dim) * (1.0 / (input_dim ** 0.5))
        )
        
        # Stack of resampler blocks (cross-attention → self-attention)
        self.blocks = nn.ModuleList([
            ResamplerBlock(
                input_dim=input_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """
        Resample vision tokens.
        
        Args:
            vision_tokens: Tensor of shape (B, N, D)
        
        Returns:
            Resampled tokens of shape (B, K, D)
        """
        B = vision_tokens.shape[0]
        latents = self.latents.expand(B, -1, -1)  # (B, K, D)
        
        # Process through resampler blocks
        for block in self.blocks:
            latents = block(latents, vision_tokens)
        
        latents = self.norm(latents)
        return latents


class ResamplerBlock(nn.Module):
    """Single Perceiver resampler block: cross-attn + self-attn."""
    
    def __init__(
        self,
        input_dim: int = 4096,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Cross-attention: latents attend to input
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(input_dim)
        
        # Self-attention on latents
        self.self_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(input_dim)
        
        # MLP
        mlp_hidden = int(input_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, input_dim),
        )
        self.norm3 = nn.LayerNorm(input_dim)
    
    def forward(self, latents: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Cross-attention
        x, _ = self.cross_attn(latents, input_tokens, input_tokens)
        latents = self.norm1(latents + x)
        
        # Self-attention
        x, _ = self.self_attn(latents, latents, latents)
        latents = self.norm2(latents + x)
        
        # MLP
        x = self.mlp(latents)
        latents = self.norm3(latents + x)
        
        return latents


if __name__ == "__main__":
    # Example usage
    compressor = VisionTokenCompressor(
        num_output_tokens=32,
        hidden_dim=4096,
    )
    
    # Dummy vision tokens (B=2, N=576, D=4096)
    vision_tokens = torch.randn(2, 576, 4096)
    
    # Compress
    compressed = compressor(vision_tokens)
    
    print(f"Input shape: {vision_tokens.shape}")
    print(f"Output shape: {compressed.shape}")
    print(f"Compression ratio: {compressor.get_compression_ratio(576):.2f}x")
