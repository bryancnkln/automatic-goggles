"""
Main Agent Entry Point

Example script showing how to initialize and run the agent.
"""

import logging
import argparse
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision import DeepSeekVisionEncoder, VisionProjector, VisionTokenCompressor
from src.memory import MemoryStore
from src.agent import LLMInterface, TaskExecutor, ScreenAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run DeepSeek Screen Agent")
    parser.add_argument(
        "--instruction",
        required=True,
        help="Task instruction (e.g., 'Save document to ~/Documents')",
    )
    parser.add_argument(
        "--projector_path",
        default="checkpoints/projector_stage_a/best_model.pt",
        help="Path to trained projector checkpoint",
    )
    parser.add_argument(
        "--memory_index_path",
        default="checkpoints/memory_index.faiss",
        help="Path to FAISS memory index",
    )
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--enable_compression", action="store_true")
    parser.add_argument(
        "--llm_model",
        default="ibm/granite-7b-1m-instruct",
        help="HuggingFace model ID for LLM",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("DeepSeek Screen Agent")
    logger.info("=" * 80)
    
    # 1. Initialize vision encoder (frozen)
    logger.info("Loading vision encoder...")
    vision_encoder = DeepSeekVisionEncoder(
        model_path="deepseek-ai/DeepSeek-OCR",
        device=args.device,
        freeze=True,
    )
    
    # 2. Load projector
    logger.info(f"Loading projector from {args.projector_path}...")
    projector = VisionProjector(in_dim=2048, out_dim=4096)
    
    if Path(args.projector_path).exists():
        checkpoint = torch.load(args.projector_path, map_location=args.device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            projector.load_state_dict(checkpoint["model_state"])
        else:
            projector.load_state_dict(checkpoint)
        logger.info("Projector loaded successfully")
    else:
        logger.warning(f"Projector not found at {args.projector_path}")
        logger.warning("Using randomly initialized projector (not recommended for inference)")
    
    projector = projector.to(args.device)
    projector.eval()
    
    # 3. Initialize compressor (optional)
    compressor = None
    if args.enable_compression:
        logger.info("Initializing token compressor...")
        compressor = VisionTokenCompressor(
            num_output_tokens=32,
            hidden_dim=4096,
        ).to(args.device)
        compressor.eval()
    
    # 4. Initialize LLM
    logger.info(f"Loading LLM: {args.llm_model}...")
    llm = LLMInterface(
        model_name=args.llm_model,
        device=args.device,
        dtype=torch.float16,
        load_in_8bit=True,
    )
    
    # 5. Initialize memory
    logger.info("Initializing memory store...")
    memory = MemoryStore(
        max_screens=1000,
        embedding_dim=4096,
        retrieval_metric="L2",
    )
    
    if Path(args.memory_index_path).exists():
        logger.info(f"Loading memory index from {args.memory_index_path}...")
        try:
            memory.load(Path(args.memory_index_path).parent)
        except Exception as e:
            logger.warning(f"Could not load memory index: {e}")
    
    # 6. Initialize task executor
    logger.info("Initializing task executor...")
    executor = TaskExecutor(use_playwright=True)
    
    # 7. Create agent
    logger.info("Creating agent...")
    agent = ScreenAgent(
        vision_encoder=vision_encoder,
        projector=projector,
        llm=llm,
        memory=memory,
        executor=executor,
        compressor=compressor,
    )
    
    # 8. Run task
    logger.info("=" * 80)
    logger.info(f"Running task: {args.instruction}")
    logger.info("=" * 80)
    
    result = agent.run_task(
        instruction=args.instruction,
        max_steps=args.max_steps,
    )
    
    # 9. Print results
    logger.info("=" * 80)
    logger.info("Task Results:")
    logger.info(f"  Success: {result['success']}")
    logger.info(f"  Steps: {result['num_steps']}")
    logger.info(f"  Total Time: {result['total_time']:.2f}s")
    logger.info("=" * 80)
    
    return 0 if result["success"] else 1


if __name__ == "__main__":
    exit(main())
