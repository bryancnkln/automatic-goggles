"""
Screen Agent Loop

Main reasoning loop for vision-based task execution with memory.
"""

import logging
import time
from typing import Dict, Optional
import numpy as np
import torch
from PIL import ImageGrab

from ..vision import DeepSeekVisionEncoder, VisionProjector, VisionTokenCompressor
from ..memory import MemoryStore
from .llm_interface import LLMInterface
from .task_executor import TaskExecutor, ActionParser

logger = logging.getLogger(__name__)


class ScreenAgent:
    """
    Vision-based screen automation agent with long-term memory.
    
    Pipeline:
        1. Capture screenshot
        2. Extract vision tokens (DeepSeek)
        3. Project to LLM space
        4. Compress tokens (optional)
        5. Retrieve similar past screens
        6. Generate decision via LLM
        7. Execute action
        8. Store in memory
    """
    
    def __init__(
        self,
        vision_encoder: DeepSeekVisionEncoder,
        projector: VisionProjector,
        llm: LLMInterface,
        memory: MemoryStore,
        executor: TaskExecutor,
        compressor: Optional[VisionTokenCompressor] = None,
    ):
        """
        Initialize the agent.
        
        Args:
            vision_encoder: Frozen DeepSeek vision encoder
            projector: Trained MLP (2048 → 4096)
            llm: Long-context LLM (Granite/Qwen)
            memory: Memory store with retrieval
            executor: Task executor (Playwright/PyAutoGUI)
            compressor: Optional vision token compressor
        """
        self.vision_encoder = vision_encoder
        self.projector = projector
        self.llm = llm
        self.memory = memory
        self.executor = executor
        self.compressor = compressor
        self.action_parser = ActionParser()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def run_task(
        self,
        instruction: str,
        max_steps: int = 20,
        step_timeout_sec: int = 30,
    ) -> Dict:
        """
        Execute a high-level task.
        
        Args:
            instruction: Task description (e.g., "Save document to ~/Documents")
            max_steps: Maximum steps before terminating
            step_timeout_sec: Timeout per step
        
        Returns:
            {success, steps, total_time, ...}
        """
        start_time = time.time()
        steps = []
        task_id = f"task_{int(start_time)}"
        
        for step_num in range(max_steps):
            logger.info(f"\n--- Step {step_num + 1}/{max_steps} ---")
            
            try:
                # 1. Capture screenshot
                screenshot = ImageGrab.grab()
                screenshot.save(f"logs/screenshots/{task_id}_step_{step_num}.png")
                
                # 2. Extract vision tokens
                vision_tokens = self._extract_vision_tokens(screenshot)  # (1, 576, 2048)
                
                # 3. Project to LLM space
                projected = self.projector(vision_tokens)  # (1, 576, 4096)
                
                # 4. Compress (optional)
                if self.compressor:
                    compressed = self.compressor(projected)  # (1, 32, 4096)
                else:
                    compressed = projected
                
                # 5. Create query embedding (mean pool)
                query_embedding = compressed.mean(dim=1).detach().cpu().numpy()  # (1, 4096)
                
                # 6. Retrieve similar past screens
                similar = self.memory.retrieve_similar(query_embedding[0], k=3)
                
                # 7. Build context
                context = self._build_context(instruction, similar)
                
                # 8. Generate decision
                action_str = self.llm.generate(
                    prompt=context,
                    vision_embeddings=compressed,
                    max_new_tokens=256,
                )
                
                # 9. Parse action
                action = self.action_parser.parse_json_action(action_str)
                if not action:
                    action = self.action_parser.parse_natural_action(action_str)
                
                logger.info(f"Generated action: {action}")
                
                # 10. Execute action
                success = self.executor.execute_action(action)
                
                # 11. Store in memory
                self.memory.add_screenshot(
                    screenshot_id=f"{task_id}_step_{step_num}",
                    timestamp=time.time(),
                    vision_tokens=vision_tokens[0].cpu().numpy(),
                    projected_embedding=query_embedding[0],
                    ocr_regions=[],  # TODO: Extract OCR
                    action=action,
                    task_context=instruction,
                    success=success,
                )
                
                steps.append({
                    "step": step_num,
                    "action": action,
                    "success": success,
                    "time": time.time() - start_time,
                })
                
                if not success:
                    logger.warning(f"Action failed at step {step_num}")
                    break
                
                time.sleep(0.5)  # Brief pause between steps
                
            except Exception as e:
                logger.error(f"Error at step {step_num}: {e}")
                break
        
        total_time = time.time() - start_time
        
        result = {
            "task_id": task_id,
            "instruction": instruction,
            "success": len(steps) > 0 and steps[-1]["success"],
            "num_steps": len(steps),
            "total_time": total_time,
            "steps": steps,
        }
        
        logger.info(f"\n=== Task Complete ===\n{result}")
        return result
    
    def _extract_vision_tokens(self, screenshot: "PIL.Image") -> torch.Tensor:
        """Extract vision tokens from screenshot."""
        # Convert PIL to tensor
        img_array = np.array(screenshot).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Forward through DeepSeek encoder
        with torch.no_grad():
            vision_tokens = self.vision_encoder(img_tensor)  # (1, 576, 2048)
        
        return vision_tokens.to(self.device)
    
    def _build_context(self, instruction: str, similar: list) -> str:
        """Build LLM context from instruction and memory."""
        context = f"Task: {instruction}\n\n"
        
        # Recent history
        recent_ctx = self.memory.get_recent_context(k=3)
        context += f"Recent actions:\n{recent_ctx}\n\n"
        
        # Similar screens
        if similar:
            context += "Similar past screens:\n"
            for i, screen in enumerate(similar):
                context += f"  [{i}] OCR: {screen['ocr_summary'][:100]} | Action: {screen['action']}\n"
            context += "\n"
        
        context += "Current screen: <VISION_TOKENS>\n"
        context += "What action should I take next? Output JSON: {\"action\": \"...\", \"target\": \"...\", \"position\": [x, y]}\n"
        
        return context
