"""
LLM Interface for Granite-7B-1M and Qwen2-7B-1M

Handles loading, inference, and embedding injection for vision tokens.
"""

import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LLMInterface:
    """
    Interface to long-context LLMs (Granite-7B-1M, Qwen2-7B-1M).
    
    Supports:
        - Direct token injection (embedding-level concatenation)
        - Vision token pooling for efficient inference
        - LoRA finetuning (optional)
    """
    
    def __init__(
        self,
        model_name: str = "ibm/granite-7b-1m-instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = True,
        load_in_4bit: bool = False,
    ):
        """
        Initialize LLM interface.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            dtype: Data type for computation
            load_in_8bit: Quantize to 8-bit
            load_in_4bit: Quantize to 4-bit
        """
        self.model_name = model_name
        # Normalize device for Apple Silicon MPS
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device
        self.dtype = dtype
        
        logger.info(f"Loading {model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        quantization_config = None
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        
        # Load with SDPA attention for optimal MPS performance
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                quantization_config=quantization_config,
                attn_implementation="sdpa",
                # For MPS we avoid device_map="auto"; we load on CPU then move to MPS
                device_map=None,
                trust_remote_code=True,
            )
        except TypeError:
            # Older transformers versions use _attn_implementation
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                quantization_config=quantization_config,
                _attn_implementation="sdpa",
                device_map=None,
                trust_remote_code=True,
            )
        # Move to device when not using accelerate's device_map
        try:
            if self.device == "mps":
                self.model = self.model.to("mps")
            elif self.device == "cuda":
                self.model = self.model.to("cuda")
            else:
                self.model = self.model.to("cpu")
        except Exception as e:
            logger.warning(f"Failed to move model to {self.device}: {e}")
        
        self.model.eval()
        self.hidden_dim = self.model.config.hidden_size
        
        logger.info(f"Loaded {model_name} (hidden_dim={self.hidden_dim})")
    
    def get_hidden_dim(self) -> int:
        """Return the hidden dimension of the LLM."""
        return self.hidden_dim
    
    def tokenize(
        self,
        text: str,
        max_length: int = 2048,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize text.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
        
        Returns:
            (input_ids, attention_mask) tensors
        """
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="longest",
            return_tensors="pt",
        )
        
        return tokens["input_ids"].to(self.device), tokens["attention_mask"].to(self.device)
    
    def get_text_embeddings(self, text: str) -> torch.Tensor:
        """
        Get text embeddings by running through embedding layer.
        
        Args:
            text: Input text
        
        Returns:
            Embeddings of shape (1, T, hidden_dim)
        """
        input_ids, _ = self.tokenize(text)
        
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(input_ids)
        
        return embeddings
    
    def generate(
        self,
        prompt: str,
        vision_embeddings: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text, optionally with vision token injection.
        
        Args:
            prompt: Text prompt with optional <image> placeholder
            vision_embeddings: (1, N, hidden_dim) optional vision tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p nucleus sampling
        
        Returns:
            Generated text
        """
        # Tokenize prompt
        input_ids, attention_mask = self.tokenize(prompt)
        
        # Get text embeddings
        text_embeddings = self.model.get_input_embeddings()(input_ids)
        
        # Inject vision embeddings if provided
        if vision_embeddings is not None:
            # Simple concatenation: [text_left, vision, text_right]
            # In practice, you'd want to handle <image> placeholder
            embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1)
            attention_mask_vision = torch.ones(
                vision_embeddings.shape[0],
                vision_embeddings.shape[1],
                device=self.device,
            )
            attention_mask = torch.cat([attention_mask_vision, attention_mask], dim=1)
        else:
            embeddings = text_embeddings
        
        # Generate using inputs_embeds instead of input_ids
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def forward_pass(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass on embedding inputs.
        
        Args:
            embeddings: (B, T, hidden_dim) input embeddings
            attention_mask: (B, T) attention mask
        
        Returns:
            Logits (B, T, vocab_size)
        """
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                return_dict=True,
            )
        
        return outputs.logits
    
    def apply_lora(self, lora_rank: int = 8, lora_alpha: int = 16) -> None:
        """
        Apply LoRA to the LLM for efficient finetuning.
        
        Args:
            lora_rank: Rank of LoRA matrices
            lora_alpha: Alpha scaling factor
        """
        try:
            from peft import get_peft_model, LoraConfig
            
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            logger.info(f"Applied LoRA (rank={lora_rank})")
        except ImportError:
            logger.error("peft not installed. Install with: pip install peft")
