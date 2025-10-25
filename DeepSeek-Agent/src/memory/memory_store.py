"""
Unified Memory Store

Combines screen history storage and dense retrieval for agent memory.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from .screen_memory import ScreenMemory
from .retrieval import DenseRetrieval

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Unified memory system combining:
        - ScreenMemory: chronological storage of screenshots
        - DenseRetrieval: FAISS index for similarity search
    """
    
    def __init__(
        self,
        max_screens: int = 1000,
        embedding_dim: int = 4096,
        retrieval_metric: str = "L2",
    ):
        """
        Initialize memory store.
        
        Args:
            max_screens: Maximum screens in history
            embedding_dim: Dimension of vision embeddings
            retrieval_metric: Distance metric for FAISS
        """
        self.screen_memory = ScreenMemory(max_screens=max_screens)
        self.retrieval_index = DenseRetrieval(
            embedding_dim=embedding_dim,
            metric=retrieval_metric,
        )
    
    def add_screenshot(
        self,
        screenshot_id: str,
        timestamp: float,
        vision_tokens: np.ndarray,
        projected_embedding: np.ndarray,  # (4096,) for embedding search
        ocr_regions: List[Dict],
        action: Dict,
        task_context: str,
        success: bool,
    ) -> None:
        """
        Add a screenshot to both storage and index.
        
        Args:
            screenshot_id: Unique ID
            timestamp: Unix timestamp
            vision_tokens: (N, 2048 or 4096) raw or projected tokens
            projected_embedding: (4096,) mean-pooled embedding for retrieval
            ocr_regions: List of OCR detections
            action: Action taken {type, target, position, args}
            task_context: Task description
            success: Whether action succeeded
        """
        # Add to chronological memory
        self.screen_memory.add_screen(
            screenshot_id=screenshot_id,
            timestamp=timestamp,
            vision_tokens=vision_tokens,
            ocr_regions=ocr_regions,
            action=action,
            task_context=task_context,
            success=success,
        )
        
        # Add to retrieval index
        metadata = {
            "screenshot_id": screenshot_id,
            "task_context": task_context,
            "ocr_summary": " ".join([r.get("text", "") for r in ocr_regions]),
            "action": action,
            "success": success,
        }
        self.retrieval_index.add(projected_embedding, metadata)
    
    def retrieve_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Retrieve similar past screens.
        
        Args:
            query_embedding: (4096,) query embedding
            k: Number of results
            threshold: Optional distance threshold
        
        Returns:
            List of {screenshot_id, ocr_summary, action, success, distance}
        """
        results = self.retrieval_index.search(query_embedding, k=k, threshold=threshold)
        
        similar_screens = []
        for idx, distance, metadata in results:
            screen_id = metadata["screenshot_id"]
            screen = self.screen_memory.get_screen(screen_id)
            
            if screen:
                similar_screens.append({
                    "screenshot_id": screen_id,
                    "ocr_summary": metadata["ocr_summary"],
                    "action": metadata["action"],
                    "success": metadata["success"],
                    "task_context": metadata["task_context"],
                    "distance": distance,
                })
        
        return similar_screens
    
    def get_recent_context(
        self,
        k: int = 5,
        max_tokens: int = 2000,
    ) -> str:
        """
        Get context from recent screenshots.
        
        Args:
            k: Number of recent screenshots
            max_tokens: Approximate token limit for context
        
        Returns:
            Formatted context string for LLM
        """
        recent = self.screen_memory.get_recent_screens(k=k)
        
        context_parts = []
        for i, screen in enumerate(recent):
            ocr_summary = self.screen_memory.get_ocr_summary(screen["screenshot_id"])
            action = screen["action"]
            success = screen["success"]
            
            part = f"[Step {i}] OCR: {ocr_summary} | Action: {action} | Success: {success}"
            context_parts.append(part)
        
        context = "\n".join(context_parts)
        
        # Simple truncation if too long
        if len(context) > max_tokens * 4:  # Rough approximation: 4 chars per token
            context = context[:max_tokens * 4] + "..."
        
        return context
    
    def get_task_history(self, task_context: str) -> str:
        """Get history for a specific task."""
        screens = self.screen_memory.get_by_task(task_context)
        
        context_parts = []
        for i, screen in enumerate(screens):
            ocr_summary = self.screen_memory.get_ocr_summary(screen["screenshot_id"])
            action = screen["action"]
            success = screen["success"]
            
            part = f"[Task Step {i}] OCR: {ocr_summary} | Action: {action} | Success: {success}"
            context_parts.append(part)
        
        return "\n".join(context_parts)
    
    def clear(self) -> None:
        """Clear all memory."""
        self.screen_memory.clear()
        self.retrieval_index.clear()
    
    def size(self) -> Dict[str, int]:
        """Return memory sizes."""
        return {
            "screens": self.screen_memory.size(),
            "indexed": self.retrieval_index.size(),
        }
    
    def save(self, save_dir: str) -> None:
        """Save entire memory store."""
        self.screen_memory.save_to_disk(save_dir)
        self.retrieval_index.save(f"{save_dir}/index.faiss")
        logger.info(f"Saved memory store to {save_dir}")
    
    def load(self, load_dir: str) -> None:
        """Load entire memory store."""
        self.screen_memory.load_from_disk(load_dir)
        self.retrieval_index.load(f"{load_dir}/index.faiss")
        logger.info(f"Loaded memory store from {load_dir}")
