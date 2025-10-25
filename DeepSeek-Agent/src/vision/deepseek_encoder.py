"""
DeepSeek Vision Encoder Wrapper

Interfaces with the frozen DeepSeek-OCR vision backbone to extract vision tokens.
This encoder is frozen and not trained; it serves as the visual input source.
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class DeepSeekVisionEncoder(nn.Module):
    """
    Wrapper for DeepSeek-OCR vision encoder.
    
    Outputs: (B, N, 2048) vision tokens per screenshot, where N is typically 576.
    This encoder remains frozen throughout training.
    """
    
    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        freeze: bool = True,
    ):
        """
        Initialize the DeepSeek vision encoder.
        
        Args:
            model_path: HuggingFace model identifier or local path
            device: Device to run encoder on ("cuda" or "cpu")
            dtype: Data type for encoder computations
            freeze: Whether to freeze encoder parameters
        """
        super().__init__()
        
        self.model_path = model_path
        # Normalize device: prefer MPS on Apple Silicon if requested and available
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device
        self.dtype = dtype
        self.freeze_params = freeze
        self.output_dim = 2048
        
        # TODO: Load the actual DeepSeek-OCR vision backbone
        # For now, we create a placeholder that matches the expected interface
        # Replace this with actual model loading when DeepSeek repo is fully integrated
        
        try:
            # Attempt to load from transformers if available
            from transformers import AutoModel
            try:
                self.encoder = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    attn_implementation="sdpa",
                )
            except TypeError:
                self.encoder = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    _attn_implementation="sdpa",
                )
            # Move to device (supports mps/cpu/cuda)
            self.encoder = self.encoder.to(self.device)
        except Exception as e:
            logger.warning(f"Could not load DeepSeek from transformers: {e}")
            logger.info("Using placeholder encoder. Wire the actual DeepSeek-OCR encoder here.")
            self.encoder = None
        
        if self.freeze_params and self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract vision tokens from images.
        
        Args:
            images: Tensor of shape (B, 3, H, W) with pixel values in [0, 1]
        
        Returns:
            Vision tokens of shape (B, N, 2048), where N ≈ 576 for standard images
        """
        if self.encoder is None:
            raise RuntimeError(
                "DeepSeek encoder not loaded. "
                "Please configure the model path and ensure the package is installed."
            )
        
        # Prefer channels_last for better memory throughput in vision
        images = images.to(self.device).to(self.dtype)
        if images.dim() == 4:
            images = images.to(memory_format=torch.channels_last)
        
        with torch.no_grad():
            # Forward through the DeepSeek encoder
            # Expected output: per-patch vision tokens
            outputs = self.encoder(images)
            
            # Extract tokens (adjust indexing based on actual model output)
            if isinstance(outputs, dict):
                vision_tokens = outputs.get("vision_tokens", outputs.get("last_hidden_state"))
            else:
                vision_tokens = outputs
        
        return vision_tokens
    
    def encode_from_file(self, image_path: str) -> torch.Tensor:
        """
        Encode image from file path.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Vision tokens of shape (1, N, 2048)
        """
        image = Image.open(image_path).convert("RGB")
        return self.encode_pil_image(image)
    
    def encode_pil_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode PIL Image to vision tokens.
        
        Args:
            image: PIL Image
        
        Returns:
            Vision tokens of shape (1, N, 2048)
        """
        # Normalize to [0, 1] if needed
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        
        return self.forward(image_tensor)
    
    def freeze(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_output_dim(self) -> int:
        """Return the output dimension of vision tokens."""
        return self.output_dim


if __name__ == "__main__":
    # Example usage
    encoder = DeepSeekVisionEncoder(
        model_path="deepseek-ai/DeepSeek-OCR",
        device="cuda" if torch.cuda.is_available() else "cpu",
        freeze=True,
    )
    
    # Create dummy image
    dummy_image = torch.randn(1, 3, 224, 224)
    tokens = encoder(dummy_image)
    print(f"Output shape: {tokens.shape}")
