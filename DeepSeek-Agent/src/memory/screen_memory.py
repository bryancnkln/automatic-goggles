"""
Screen Memory Manager

Stores and retrieves past screenshots, vision tokens, OCR regions, and
actions for memory-augmented agent reasoning.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ScreenMemory:
    """
    In-process memory for storing screenshot history and metadata.
    
    Each entry contains:
        - screenshot_id: unique identifier
        - timestamp: when the screenshot was captured
        - vision_tokens: np.ndarray (N, 2048) or (N, 4096) after projection
        - ocr_regions: list of {text, bbox, confidence}
        - action: {type, target, position, args}
        - task_context: high-level task description
        - success: whether action succeeded
    """
    
    def __init__(self, max_screens: int = 1000):
        """
        Initialize screen memory.
        
        Args:
            max_screens: Maximum number of screens to keep (FIFO eviction)
        """
        self.max_screens = max_screens
        self.memory: List[Dict] = []
        self.screen_ids: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None  # Will be populated during indexing
    
    def add_screen(
        self,
        screenshot_id: str,
        timestamp: float,
        vision_tokens: np.ndarray,
        ocr_regions: List[Dict],
        action: Dict,
        task_context: str,
        success: bool,
    ) -> None:
        """
        Add a screenshot to memory.
        
        Args:
            screenshot_id: Unique identifier for this screenshot
            timestamp: Unix timestamp
            vision_tokens: (N, 2048) or (N, 4096) vision token array
            ocr_regions: List of {text, bbox, confidence}
            action: {type, target, position, args}
            task_context: Task description
            success: Whether action succeeded
        """
        entry = {
            "screenshot_id": screenshot_id,
            "timestamp": timestamp,
            "vision_tokens": vision_tokens,
            "ocr_regions": ocr_regions,
            "action": action,
            "task_context": task_context,
            "success": success,
        }
        
        self.memory.append(entry)
        self.screen_ids.append(screenshot_id)
        
        # FIFO eviction if exceeds max
        if len(self.memory) > self.max_screens:
            self.memory.pop(0)
            self.screen_ids.pop(0)
        
        logger.debug(f"Added screen {screenshot_id} to memory (total: {len(self.memory)})")
    
    def get_screen(self, screenshot_id: str) -> Optional[Dict]:
        """Retrieve a specific screenshot by ID."""
        for entry in self.memory:
            if entry["screenshot_id"] == screenshot_id:
                return entry
        return None
    
    def get_all_screens(self) -> List[Dict]:
        """Return all screens in memory."""
        return self.memory
    
    def get_recent_screens(self, k: int = 5) -> List[Dict]:
        """Return the k most recent screens."""
        return self.memory[-k:]
    
    def get_successful_screens(self) -> List[Dict]:
        """Return only screens where action was successful."""
        return [e for e in self.memory if e.get("success", False)]
    
    def get_by_task(self, task_context: str) -> List[Dict]:
        """Return screens for a specific task."""
        return [e for e in self.memory if e.get("task_context") == task_context]
    
    def get_ocr_summary(self, screenshot_id: str) -> str:
        """Get concatenated OCR text from a screenshot."""
        screen = self.get_screen(screenshot_id)
        if not screen:
            return ""
        ocr_texts = [r.get("text", "") for r in screen.get("ocr_regions", [])]
        return " ".join(ocr_texts)
    
    def clear(self) -> None:
        """Clear all memory."""
        self.memory.clear()
        self.screen_ids.clear()
        self.embeddings = None
        logger.info("Cleared screen memory")
    
    def size(self) -> int:
        """Return number of screens in memory."""
        return len(self.memory)
    
    def save_to_disk(self, save_dir: str) -> None:
        """
        Save memory to disk for persistence.
        
        Saves:
            - metadata.jsonl: All metadata (OCR, actions, etc.)
            - tokens/: Vision tokens as .npy files
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        tokens_dir = save_path / "tokens"
        tokens_dir.mkdir(exist_ok=True)
        
        # Save metadata
        with open(save_path / "metadata.jsonl", "w") as f:
            for entry in self.memory:
                # Don't save vision tokens in JSONL (too large)
                metadata = {
                    "screenshot_id": entry["screenshot_id"],
                    "timestamp": entry["timestamp"],
                    "num_tokens": entry["vision_tokens"].shape[0],
                    "ocr_regions": entry["ocr_regions"],
                    "action": entry["action"],
                    "task_context": entry["task_context"],
                    "success": entry["success"],
                }
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                
                # Save vision tokens
                token_path = tokens_dir / f"{entry['screenshot_id']}.npy"
                np.save(token_path, entry["vision_tokens"])
        
        logger.info(f"Saved {len(self.memory)} screens to {save_dir}")
    
    def load_from_disk(self, load_dir: str) -> None:
        """
        Load memory from disk.
        
        Args:
            load_dir: Directory containing metadata.jsonl and tokens/
        """
        load_path = Path(load_dir)
        tokens_dir = load_path / "tokens"
        
        if not (load_path / "metadata.jsonl").exists():
            logger.warning(f"No metadata.jsonl found in {load_dir}")
            return
        
        with open(load_path / "metadata.jsonl") as f:
            for line in f:
                metadata = json.loads(line)
                screenshot_id = metadata["screenshot_id"]
                
                # Load vision tokens
                token_path = tokens_dir / f"{screenshot_id}.npy"
                if token_path.exists():
                    vision_tokens = np.load(token_path)
                else:
                    logger.warning(f"Missing tokens for {screenshot_id}")
                    continue
                
                self.add_screen(
                    screenshot_id=screenshot_id,
                    timestamp=metadata["timestamp"],
                    vision_tokens=vision_tokens,
                    ocr_regions=metadata["ocr_regions"],
                    action=metadata["action"],
                    task_context=metadata["task_context"],
                    success=metadata["success"],
                )
        
        logger.info(f"Loaded {len(self.memory)} screens from {load_dir}")
