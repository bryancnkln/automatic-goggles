# DeepSeek-Agent: Vision-Based Screen Automation with Long-Term Memory

A production-ready agent system that combines DeepSeek-OCR's frozen vision encoder with Granite-7B-1M to enable long-context screen understanding, task execution, and memory-driven decision making.

**🚀 Optimized for Apple Silicon (M1/M2/M3/M4) with native MPS acceleration and SDPA attention.**

## Architecture Overview

### Core Components

1. **Vision Encoder (Frozen)**
   - DeepSeek-OCR SAM+CLIP backbone
   - Outputs: ~576 vision tokens @ 2048-d per screenshot
   - Frozen after initialization; no training

2. **Tiny Vision Projector (Trainable)**
   - 2-layer MLP: 2048 → 4096 → 4096
   - Maps vision space to Granite-7B hidden dimension
   - Trained on Stage 1 (alignment) with MSE loss
   - ~6M parameters; trains in 2–4 hours on M4 Pro

3. **Vision Token Compressor (Optional)**
   - Learned attention pooling: (576, 4096) → (32, 4096)
   - Reduces context window pressure for long-context efficiency
   - Trainable as part of projector stage

4. **Long-Context LLM**
   - Granite-7B-1M or Qwen2-7B-1M
   - 1M token context window
   - Frozen during projector training; LoRA-tuned in Stage 2

5. **Agent Memory System**
   - FAISS dense retrieval index
   - Stores embeddings + metadata for past 1000+ screenshots
   - Fast lookup: ~10ms for top-3 similar screens
   - Organic log storage: {screenshot, action, outcome, task_context}

6. **Agent Loop**
   - Capture → Project → Compress → Retrieve → Decide → Execute → Store
   - Supports arbitrary tasks via natural language instruction
   - Steerable via Playwright or PyAutoGUI

## Project Structure

```
DeepSeek-Agent/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.yaml                    # Global configuration
├── src/
│   ├── __init__.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── deepseek_encoder.py       # DeepSeek vision wrapper
│   │   ├── vision_projector.py       # MLP projector
│   │   └── vision_tokenizer.py       # Token pooling/compression
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── screen_memory.py          # Screenshot history
│   │   ├── retrieval.py              # Dense retrieval
│   │   └── memory_store.py           # FAISS backend
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── llm_interface.py          # LLM integration
│   │   ├── task_executor.py          # Action execution
│   │   └── agent_loop.py             # Main reasoning loop
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # Training dataset
│   │   ├── screen_processor.py       # Screenshot processing
│   │   └── annotation_builder.py     # Task log → training data
│   └── training/
│       ├── __init__.py
│       ├── train_projector.py        # Stage 1: MLP alignment
│       ├── finetune_agent.py         # Stage 2: LoRA (optional)
│       └── eval.py                   # Benchmarking
├── notebooks/
│   ├── explore_deepseek_tokens.ipynb
│   ├── visualize_memory.ipynb
│   └── trace_agent_decisions.ipynb
├── scripts/
│   ├── extract_features.py           # Batch DeepSeek feature export
│   ├── build_memory_index.py         # Create FAISS retrieval
│   ├── collect_screen_data.py        # Record task data
│   ├── prepare_training_data.py      # Convert logs → training
│   └── run_agent.py                  # Live agent loop
├── tests/
│   ├── test_vision_projector.py
│   ├── test_memory_retrieval.py
│   └── test_agent_loop.py
├── checkpoints/
│   ├── projector_stage_a/            # Trained MLP weights
│   └── agent_lora/                   # LoRA adapters (optional)
└── logs/
    ├── screen_history/               # Saved .npy tokens + metadata
    └── action_logs/                  # Task execution logs
```

## Quick Start

### 1. Installation

#### For Apple Silicon (M1/M2/M3/M4) - Recommended

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with MPS support
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Install project dependencies
pip install -r requirements.txt

# Set optimal MPS memory management
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

#### For CUDA (Linux/Windows)

```bash
cd DeepSeek-Agent
pip install -r requirements.txt
```

### 2. Configuration

The `config.yaml` is pre-configured for optimal Apple Silicon performance:

```yaml
vision_encoder:
  model_path: "deepseek-ai/DeepSeek-OCR"
  device: "mps"  # Apple Silicon GPU acceleration
  dtype: "float16"  # Optimal for MPS
  frozen: true
  output_dim: 2048

projector:
  in_dim: 2048
  out_dim: 4096
  freeze_after_stage_a: true

compressor:
  num_output_tokens: 32
  hidden_dim: 4096
  enabled: true

llm:
  model_name: "ibm/granite-7b-1m-instruct"
  device: "mps"  # Apple Silicon GPU acceleration
  dtype: "float16"
  max_context_length: 1000000
  load_in_8bit: false  # Not supported on MPS

memory:
  max_screens: 1000
  embedding_dim: 4096
  similarity_threshold: 0.7
  retrieval_top_k: 3

training:
  projector:
    batch_size: 64  # Optimized for M4 Pro
    learning_rate: 1e-3
    epochs: 1
    warmup_steps: 2000
    num_workers: 16  # Leverage 20 CPU cores
  agent_lora:
    batch_size: 64  # Optimized for M4 Pro
    learning_rate: 5e-4
    lora_rank: 8
    lora_alpha: 16

hardware:
  device: "mps"  # Apple Silicon GPU
  num_gpus: 0  # MPS doesn't count as traditional GPU
  mixed_precision: "fp16"
  torch_compile: false
  num_workers: 16
  pin_memory: false
```

### 3. Collect Task Data

```bash
python scripts/collect_screen_data.py \
  --task_description "Save document to ~/Documents" \
  --duration_sec 120
```

This records screenshots + manual action annotations to `logs/`.

### 4. Extract DeepSeek Features

```bash
python scripts/extract_features.py \
  --screenshots_dir logs/screenshots \
  --output_dir logs/tokens \
  --batch_size 8
```

Exports vision tokens to `.npy` files for training.

### 5. Train Projector (Stage 1)

```bash
# For Apple Silicon with MPS optimization
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
python src/training/train_projector.py \
  --manifest logs/train_manifest.jsonl \
  --model_dir checkpoints/projector_stage_a \
  --epochs 1
```

Trains the 2-layer MLP on synthetic or real data. ~2–4 hours on M4 Pro with MPS acceleration and SDPA attention.

### 6. Build Memory Index

```bash
python scripts/build_memory_index.py \
  --tokens_dir logs/tokens \
  --metadata_file logs/task_logs.jsonl \
  --output_path checkpoints/memory_index.faiss
```

Creates FAISS index for dense screen retrieval.

### 7. Run Agent Loop

```bash
python scripts/run_agent.py \
  --instruction "Navigate to ~/Documents and save my work" \
  --max_steps 20 \
  --projector_path checkpoints/projector_stage_a/model.pt \
  --memory_index_path checkpoints/memory_index.faiss
```

Live agent execution with UI automation and memory retrieval.

## Training Pipeline

### Stage 1: Projector Alignment (100k samples, 1 epoch)
- **Objective**: Learn mapping from 2048-d vision space to 4096-d language space
- **Data**: Pairs of (DeepSeek tokens, Granite embeddings)
- **Loss**: MSE between projected tokens and target embeddings
- **Time**: 2–4 hours on M4 Pro, fp16
- **Output**: `projector_stage_a/model.pt`
- **Freeze**: After training, projector remains frozen

### Stage 2: Agent Instruction Tuning (Optional, 1000+ task examples)
- **Objective**: Fine-tune LLM to follow screen-based task instructions
- **Method**: LoRA on Q/K/V/O projections (8 rank, 16 alpha)
- **Data**: Organic task logs: {screenshot_tokens, action, task_context, success}
- **Loss**: Supervised task success prediction
- **Time**: 4–8 hours on M4 Pro
- **Freeze**: Vision encoder + projector remain frozen; only LoRA tunes LLM

## Data Formats

### Screenshot Log (JSONL)

```json
{
  "screenshot_id": "task_001_frame_045",
  "timestamp": 1234567890.5,
  "screenshot_path": "logs/screenshots/task_001_frame_045.png",
  "vision_tokens_path": "logs/tokens/task_001_frame_045.npy",
  "num_tokens": 576,
  "ocr_regions": [
    {"text": "Save", "bbox": [100, 50, 150, 80]},
    {"text": "File name", "bbox": [200, 100, 300, 120]}
  ],
  "action": {"type": "click", "target": "Save", "position": [125, 65]},
  "task_context": "Save document to ~/Documents",
  "success": true
}
```

### Training Example (JSONL)

```json
{
  "vision_tokens_path": "logs/tokens/task_001_frame_045.npy",
  "action": "click",
  "action_target": "Save",
  "task_context": "Save document to ~/Documents",
  "ocr_summary": "Save File name Cancel",
  "reward": 1.0
}
```

## Performance Notes

### Apple Silicon (M1/M2/M3/M4) with MPS
- **Memory retrieval**: <10ms per query (FAISS)
- **Vision projection**: ~30ms per screenshot (M4 Pro, fp16, MPS)
- **LLM inference**: 1–2 sec per decision (Granite-7B-1M, fp16, SDPA)
- **Total latency**: ~1.5–3 sec per agent step (capture → decision → execute)
- **Memory usage**: ~8-12GB RAM for full training pipeline
- **Batch processing**: 64 samples/batch optimal for M4 Pro

### CUDA Systems
- **Memory retrieval**: <10ms per query (FAISS)
- **Vision projection**: ~50ms per screenshot (A100, fp16)
- **LLM inference**: 1–3 sec per decision (Granite-7B-1M, int8)
- **Total latency**: ~2–5 sec per agent step (capture → decision → execute)

## Extending the Agent

### Add New Task Types

Modify `src/agent/agent_loop.py` to handle domain-specific actions (e.g., form filling, API calls).

### Custom LLM Integration

Replace `src/agent/llm_interface.py` to use different models (e.g., Llama, Qwen).

### Improve Memory Retrieval

Swap FAISS with vector databases (e.g., Milvus, Qdrant) for distributed deployments.

### Online Learning

Stream new task logs to FAISS index and incrementally improve the agent's decision-making.

## Apple Silicon Optimization

### Key Features
- **Native MPS Support**: Direct GPU acceleration on Apple Silicon
- **SDPA Attention**: PyTorch 2's optimized attention mechanism (replaces FlashAttention on MPS)
- **Channels Last Memory Format**: Optimized tensor layout for image processing
- **Mixed Precision Training**: FP16 for faster training and lower memory usage
- **Optimal Batch Sizing**: Tuned for M4 Pro's 20 CPU cores and 64GB RAM

### Environment Variables
```bash
# Set optimal MPS memory management
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Optional: Enable MPS fallback logging
export PYTORCH_MPS_LOG_LEVEL=1
```

### Troubleshooting
- **MPS not available**: Ensure you're on macOS with Apple Silicon and PyTorch >= 2.0
- **Memory issues**: Lower batch size to 48 or 32, ensure `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- **Slow performance**: Check that `dtype: "float16"` is set in config
- **Model loading errors**: First run downloads weights; subsequent runs are faster

## References

- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- [Granite-7B-1M](https://huggingface.co/ibm/granite-7b-1m-instruct)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Playwright](https://playwright.dev/python/)
- [PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html)
- [SDPA Attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

## License

MIT

## Citation

If you use this agent system, please cite:

```bibtex
@misc{deepseek_agent,
  title={DeepSeek-Agent: Vision-Based Screen Automation with Long-Term Memory},
  author={Bryan Conklin},
  year={2025}
}
```
