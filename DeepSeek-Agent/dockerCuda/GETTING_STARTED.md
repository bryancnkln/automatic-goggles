# Getting Started with DeepSeek-Agent

This guide walks you through setting up and running the DeepSeek-Agent system.

## Prerequisites

- **Hardware**: M4 Pro with 16GB+ unified memory (or GPU with 24GB+ VRAM)
- **OS**: macOS 12+, Linux, or Windows with WSL2
- **Python**: 3.10+

## 1. Installation

```bash
cd DeepSeek-Agent
pip install -r requirements.txt
```

## 2. Quick Configuration

Edit `config.yaml` with your setup:

```yaml
hardware:
  device: "cuda"  # or "mps" for Apple Silicon

llm:
  model_name: "ibm/granite-7b-1m-instruct"  # or Qwen2-7B-1M
  load_in_8bit: true

training:
  projector:
    batch_size: 32
    learning_rate: 1e-3
    epochs: 1
```

## 3. Stage 1: Train the Projector (One-time Setup)

### Step 1a: Prepare Training Data

If you don't have training data, start with synthetic pairs or download COCO:

```bash
# Create dummy training manifest
python scripts/prepare_training_data.py \
  --output logs/train_manifest.jsonl \
  --num_samples 100
```

### Step 1b: Train Projector

```bash
python -m src.training.train_projector \
  --manifest logs/train_manifest.jsonl \
  --save_dir checkpoints/projector_stage_a \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --epochs 1
```

**Expected**: ~2-4 hours on M4 Pro, produces `checkpoints/projector_stage_a/best_model.pt`

## 4. Collect Task Data (Optional but Recommended)

For better performance, collect real screenshots + actions:

```bash
python scripts/collect_screen_data.py \
  --task_description "Save document to ~/Documents" \
  --duration_sec 120
```

This saves to `logs/screenshots/` and `logs/metadata.jsonl`.

## 5. Build Memory Index

```bash
python scripts/build_memory_index.py \
  --tokens_dir logs/tokens \
  --metadata_file logs/metadata.jsonl \
  --output_path checkpoints/memory_index.faiss
```

## 6. Run the Agent

### Example 1: Simple Task

```bash
python scripts/run_agent.py \
  --instruction "Navigate to google.com and search for Deepseek" \
  --max_steps 10 \
  --projector_path checkpoints/projector_stage_a/best_model.pt
```

### Example 2: With Compression

```bash
python scripts/run_agent.py \
  --instruction "Open a document and save it" \
  --max_steps 15 \
  --enable_compression \
  --projector_path checkpoints/projector_stage_a/best_model.pt
```

### Example 3: With Custom LLM

```bash
python scripts/run_agent.py \
  --instruction "Navigate file system and find all PDFs" \
  --llm_model "Qwen/Qwen2-7B-Instruct" \
  --max_steps 20
```

## 7. Advanced: Optional Stage 2 Finetuning

If you collect 100+ task examples with outcomes, finetune the LLM:

```bash
python src/training/finetune_agent.py \
  --task_data logs/task_training_data.jsonl \
  --projector_path checkpoints/projector_stage_a/best_model.pt \
  --lora_rank 8 \
  --lora_alpha 16 \
  --epochs 2
```

This applies LoRA to LLM Q/K/V/O projections while keeping the base frozen.

## File Structure

```
DeepSeek-Agent/
├── src/
│   ├── vision/          # Vision encoder, projector, compression
│   ├── memory/          # Screenshot history, retrieval, FAISS
│   ├── agent/           # LLM interface, task executor, agent loop
│   ├── data/            # Dataset utilities
│   └── training/        # Stage 1 & 2 training scripts
├── scripts/
│   ├── run_agent.py                    # Main entry point
│   ├── collect_screen_data.py          # Record task data
│   ├── extract_features.py             # Batch DeepSeek feature export
│   ├── build_memory_index.py           # Create FAISS index
│   └── prepare_training_data.py        # Generate training manifest
├── logs/
│   ├── screenshots/                     # Raw images
│   ├── tokens/                          # Extracted vision tokens (.npy)
│   └── metadata.jsonl                   # OCR, actions, outcomes
├── checkpoints/
│   ├── projector_stage_a/              # Trained MLP weights
│   ├── memory_index.faiss              # FAISS retrieval index
│   └── agent_lora/                     # LoRA weights (if Stage 2)
├── config.yaml                         # Configuration
├── requirements.txt                    # Dependencies
└── README.md                           # Full documentation
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in config.yaml
- Enable `load_in_8bit` or `load_in_4bit` for LLM
- Use `--enable_compression` at inference to reduce tokens

### Slow Inference

- Enable token compression: `--enable_compression`
- Reduce `memory.max_screens` in config.yaml
- Use smaller LLM: `Qwen/Qwen2-1.5B-Instruct`

### Action Execution Fails

- Ensure Playwright is installed: `pip install playwright && playwright install`
- Fallback to PyAutoGUI (automatic if Playwright unavailable)
- Check action JSON format in logs

### Poor Agent Decisions

- Collect more task examples
- Train projector on domain-specific data (Stage 1)
- Apply LoRA finetuning (Stage 2)
- Increase `max_steps` to allow recovery

## API Examples

### Minimal Agent Setup

```python
from src.vision import DeepSeekVisionEncoder, VisionProjector
from src.memory import MemoryStore
from src.agent import LLMInterface, TaskExecutor, ScreenAgent

# Initialize components
encoder = DeepSeekVisionEncoder(freeze=True)
projector = VisionProjector()
projector.load_state_dict(torch.load("checkpoints/projector_stage_a/best_model.pt"))

llm = LLMInterface(model_name="ibm/granite-7b-1m-instruct")
memory = MemoryStore(max_screens=1000)
executor = TaskExecutor()

# Create agent
agent = ScreenAgent(encoder, projector, llm, memory, executor)

# Run task
result = agent.run_task("Save document to ~/Desktop", max_steps=15)
print(result)
```

### Use Custom LLM

```python
llm = LLMInterface(
    model_name="Qwen/Qwen2-7B-Instruct",
    dtype=torch.bfloat16,
    load_in_8bit=True,
)
```

### Enable Memory Persistence

```python
memory = MemoryStore()
memory.load("checkpoints/memory_store")  # Load previous sessions
agent = ScreenAgent(..., memory=memory)

# After running tasks
memory.save("checkpoints/memory_store")
```

## Performance Benchmarks

| Component | Time (M4 Pro) | Details |
|-----------|---------------|---------|
| Vision projection | ~50ms | Per screenshot, fp16 |
| Token compression | ~20ms | 576→32 tokens |
| FAISS retrieval | ~5ms | Top-3 similar screens |
| LLM generation | 1-3s | 256 new tokens, int8 |
| **Total per step** | **2-5s** | Capture→Decide→Execute |

## Next Steps

1. **Collect domain data**: Run `collect_screen_data.py` on your task type
2. **Train projector**: `python -m src.training.train_projector` with your data
3. **Finetune agent**: Optional Stage 2 with `finetune_agent.py`
4. **Deploy**: Integrate into your application via Python API
5. **Monitor**: Log agent decisions and failures for continuous improvement

## Support & Citation

If you use DeepSeek-Agent, please cite:

```bibtex
@software{deepseek_agent_2025,
  title={DeepSeek-Agent: Vision-Based Screen Automation with Long-Term Memory},
  author={Your Name},
  year={2025}
}
```

For issues, see [GitHub Issues](https://github.com/yourusername/DeepSeek-Agent/issues).
