# DeepSeek-Agent Training Roadmap: GUI-Actor Dataset

## 🎯 Vision

Transform DeepSeek-Agent into a state-of-the-art screen automation system by training on the **1 million sample GUI-Actor dataset**, leveraging Apple Silicon MPS acceleration for optimal performance on M4 Pro hardware.

---

## 📊 Dataset Overview

### GUI-Actor (1M Samples)

**Sources**:
- **UGround** (300k): Visual grounding on web UIs
- **GUIEnv** (200k): GUI environment interactions
- **GUIAct** (200k): Web-based action sequences
- **AMEX** (150k): Financial application workflows
- **AndroidControl** (100k): Mobile app interactions
- **Wave-UI** (50k): Desktop UI automation

**Key Characteristics**:
- ✅ Coordinate-free grounding (attention-based actions)
- ✅ Multi-modal: screenshots + text + bounding boxes
- ✅ Diverse platforms: web, mobile, desktop
- ✅ Production-ready quality
- ✅ HuggingFace hosted

**Comparison with Alternatives**:

| Dataset | Samples | Focus | Coordinate | Source |
|---------|---------|-------|-----------|--------|
| **GUI-Actor** | 1.0M | Diverse UIs | ✅ Attention-based | [HuggingFace](https://huggingface.co/datasets/cckevinn/GUI-Actor-Data) |
| GUI-World | 12k videos | Dynamic evaluation | ❌ Manual | Research |
| GUICourse | 10.7M pairs | Training focus | ❌ Text-based | Research |
| GUI Odyssey | 7.7k episodes | Mobile nav | ❌ Text coords | Research |

**Why GUI-Actor?** Perfect balance of scale, quality, and diversity for training screen automation agents.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              DeepSeek-Agent Training Pipeline               │
└─────────────────────────────────────────────────────────────┘

Stage 1: Vision Alignment (24-48 hours)
  ┌─────────────────────────────────────────────┐
  │ GUI-Actor Screenshots (1M)                  │
  │         ↓                                   │
  │ DeepSeek-OCR Vision Encoder (Frozen)       │
  │ - 576 vision tokens @ 2048-d per screenshot│
  │         ↓                                   │
  │ Tiny Vision Projector (Trainable)          │
  │ - 2-layer MLP: 2048 → 4096 → 4096         │
  │ - Loss: MSE between projected & targets    │
  │         ↓                                   │
  │ Output: projector_gui_actor/model.pt       │
  └─────────────────────────────────────────────┘

Stage 2: Task Understanding (Optional, 12-24 hours)
  ┌─────────────────────────────────────────────┐
  │ Trained Projector + Granite-7B-1M          │
  │         ↓                                   │
  │ LoRA Fine-tuning on Task Context           │
  │ - Instruction tuning: "Navigate & save"    │
  │ - Task success prediction                  │
  │         ↓                                   │
  │ Output: agent_lora_gui_actor/ adapters    │
  └─────────────────────────────────────────────┘

Stage 3: Deployment (Production)
  ┌─────────────────────────────────────────────┐
  │ Frozen Vision + Trained Projector +         │
  │ Granite-7B-1M (+ optional LoRA)            │
  │         ↓                                   │
  │ Real-time Screen Automation                │
  │ - Screenshot → tokens → decision → action  │
  │ - Latency: 1.5-3 sec per step              │
  └─────────────────────────────────────────────┘
```

---

## ⚙️ Setup Instructions

### Phase 1: Environment Setup (5 minutes)

```bash
# Navigate to project
cd /path/to/DeepSeek-OCR

# Create Python environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with MPS
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Install project dependencies
pip install -r requirements.txt
pip install -r DeepSeek-Agent/requirements.txt

# Set MPS optimization
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

**Verification**:
```bash
python << 'EOF'
import torch
print(f"✅ Torch: {torch.__version__}")
print(f"✅ MPS available: {torch.backends.mps.is_available()}")
EOF
```

### Phase 2: Dataset Preparation (30-60 minutes + download time)

```bash
# Install HuggingFace tools
pip install huggingface-hub

# Download dataset (~100GB)
python << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="cckevinn/GUI-Actor-Data",
    repo_type="dataset",
    local_dir="data/gui-actor"
)
EOF

# Convert to training format
python DeepSeek-Agent/scripts/prepare_gui_actor_data.py \
  --input_json "data/gui-actor/*/aguvis_bbox*.json" \
  --images_base data/gui-actor \
  --output_manifest DeepSeek-Agent/logs/gui_actor_manifest.jsonl

# Extract vision features
python DeepSeek-Agent/scripts/extract_features.py \
  --screenshots_dir data/gui-actor \
  --output_dir DeepSeek-Agent/logs/gui_actor_tokens \
  --batch_size 16 \
  --device mps
```

### Phase 3: Stage 1 - Vision Alignment (24-48 hours)

```bash
# Train vision projector on GUI-Actor
python DeepSeek-Agent/src/training/train_projector.py \
  --manifest DeepSeek-Agent/logs/gui_actor_manifest.jsonl \
  --model_dir DeepSeek-Agent/checkpoints/projector_gui_actor \
  --epochs 2 \
  --batch_size 64 \
  --num_workers 16 \
  --device mps \
  --dtype float16 \
  --learning_rate 1e-3 \
  --warmup_steps 5000 \
  --save_interval 5000

# Monitor progress
watch -n 5 'tail -20 DeepSeek-Agent/checkpoints/projector_gui_actor/training.log'
```

**Expected Results**:
- Loss: 0.5 → 0.1
- Validation accuracy: ~85%
- GPU memory: 8-12GB
- Throughput: 200-400 samples/sec

### Phase 4: Stage 2 - Task Understanding (12-24 hours, optional)

```bash
# Fine-tune LLM with LoRA
python DeepSeek-Agent/src/training/finetune_agent.py \
  --manifest DeepSeek-Agent/logs/gui_actor_manifest.jsonl \
  --projector_path DeepSeek-Agent/checkpoints/projector_gui_actor/model.pt \
  --lora_dir DeepSeek-Agent/checkpoints/agent_lora_gui_actor \
  --batch_size 32 \
  --num_workers 16 \
  --epochs 3 \
  --device mps \
  --lora_rank 16 \
  --lora_alpha 32 \
  --learning_rate 5e-4
```

---

## 📈 Performance Targets

### On M4 Pro (20 CPU cores, 64GB RAM, 16 GPU cores)

| Stage | Operation | Speed | Memory | Time (1M samples) |
|-------|-----------|-------|--------|------------------|
| 0 | Feature extraction | 500-1000/min | 6GB | ~30 hours |
| 1 | Projector training | 200-400/sec | 10GB | ~24-48 hours |
| 2 | LoRA fine-tuning | 100-200/sec | 8GB | ~12-24 hours |
| Inference | Per-screenshot | 30-50ms | 4GB | - |

**Total Pipeline**: ~66-102 hours (~3-4 days continuous, or ~1 week with breaks)

---

## 🔧 Configuration

### config.yaml Optimized for M4 Pro

```yaml
vision_encoder:
  model_path: "deepseek-ai/DeepSeek-OCR"
  device: "mps"
  dtype: "float16"
  frozen: true
  output_dim: 2048

llm:
  model_name: "ibm/granite-7b-1m-instruct"
  device: "mps"
  dtype: "float16"
  load_in_8bit: false  # Not supported on MPS

training:
  projector:
    batch_size: 64              # M4 Pro optimized
    num_workers: 16             # Leverage 20 CPU cores
    learning_rate: 1e-3
    epochs: 2
    warmup_steps: 5000
    gradient_accumulation: 1
    save_interval: 5000
    
  agent_lora:
    batch_size: 32
    num_workers: 16
    learning_rate: 5e-4
    epochs: 3
    lora_rank: 16
    lora_alpha: 32

hardware:
  device: "mps"
  mixed_precision: "fp16"
  num_workers: 16
  pin_memory: false
  max_memory: "48GB"
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `DATASET_INTEGRATION.md` | Detailed dataset structure and conversion |
| `GUI_ACTOR_QUICKSTART.md` | Quick reference for training pipeline |
| `LOCAL_TRAINING.md` | macOS MPS setup and optimization tips |
| `README.md` | Architecture and system overview |
| `TRAINING_ROADMAP.md` | This file - strategic overview |

---

## 🚀 Deployment Checklist

- [ ] **Phase 1**: Environment setup (`.venv`, PyTorch, MPS)
- [ ] **Phase 2**: Download & prepare dataset (1.2M manifest entries)
- [ ] **Phase 3a**: Extract vision features (30 hours)
- [ ] **Phase 3b**: Stage 1 projector training (24-48 hours)
- [ ] **Phase 4**: (Optional) Stage 2 LoRA tuning (12-24 hours)
- [ ] **Validation**: Test inference on sample tasks
- [ ] **Deployment**: Integrate trained models into agent loop
- [ ] **Monitoring**: Track performance on screen automation tasks

---

## 🎓 Expected Outcomes

### Post-Training Capabilities

1. **Vision Understanding**
   - 576 vision tokens per screenshot (2048-d)
   - Projected to 4096-d language space
   - ~85% alignment accuracy with language embeddings

2. **Task Execution**
   - Understand natural language instructions
   - Identify target UI elements from screenshots
   - Generate appropriate actions (click, type, scroll)
   - Learn from task success/failure

3. **Performance**
   - Latency: 1.5-3 sec per agent step
   - Memory: 8-12GB for inference
   - Throughput: 20-30 screen interactions/minute
   - Accuracy: Expected 70-80% on ScreenSpot benchmarks

### Comparison with State-of-the-Art

| Model | Backbone | ScreenSpot-Pro | M4 Pro Speed |
|-------|----------|----------------|-------------|
| UI-TARS-7B | Qwen2-VL | 35.7 | ~500ms/step |
| GUI-Actor-7B | Qwen2-VL | **40.7** | ~50ms/step* |
| Our Agent | Granite-7B-1M | ~38-42 | ~50-100ms/step |

*With MPS optimization and SDPA attention

---

## 💡 Advanced Topics

### Data Augmentation

```bash
# Extract high-confidence samples only
python scripts/analyze_gui_actor_data.py \
  --manifest logs/gui_actor_manifest.jsonl \
  --filter_successful_only \
  --output logs/gui_actor_filtered.jsonl
```

### Curriculum Learning

```bash
# Stage training: simple → complex
# 1. Easy: single-click tasks (100k samples)
# 2. Medium: multi-step sequences (400k samples)
# 3. Hard: complex workflows (500k samples)
```

### Multi-Resolution Training

```yaml
# Support multiple screen resolutions
image_processor:
  resolutions: [640, 1024, 1280, 1920]
  enable_dynamic_resolution: true
```

---

## 🔗 Resources

- **Dataset**: [GUI-Actor on HuggingFace](https://huggingface.co/datasets/cckevinn/GUI-Actor-Data)
- **Paper**: [GUI-Actor: Rethinking GUI Interaction](https://arxiv.org/abs/2506.03143)
- **Code**: [microsoft/GUI-Actor](https://github.com/microsoft/GUI-Actor)
- **PyTorch MPS**: [PyTorch MPS Docs](https://pytorch.org/docs/stable/notes/mps.html)
- **SDPA**: [Scaled Dot Product Attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

---

## 📞 Support & Troubleshooting

### Common Issues

**Q: Training slower than expected**
- A: Verify MPS: `torch.backends.mps.is_available()`
- A: Check `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` is set
- A: Monitor GPU: `metal_trace_tool` (requires Xcode)

**Q: Out of memory errors**
- A: Reduce batch_size from 64 → 48 → 32
- A: Use gradient_accumulation to simulate larger batches
- A: Process dataset in chunks

**Q: NaN loss during training**
- A: Reduce learning_rate from 1e-3 → 5e-4
- A: Increase warmup_steps from 5000 → 10000
- A: Check for corrupted data in manifest

---

## 🎯 Success Criteria

| Milestone | Target | Status |
|-----------|--------|--------|
| Environment setup | Day 0 | ✅ |
| Dataset downloaded | Day 1 | ⏳ |
| Features extracted | Day 2-3 | ⏳ |
| Stage 1 complete | Day 3-4 | ⏳ |
| Stage 2 complete | Day 4-5 | ⏳ |
| Inference validated | Day 5 | ⏳ |
| Production ready | Day 5-6 | ⏳ |

---

**Last Updated**: October 2025  
**Target Hardware**: Apple Silicon M4 Pro (20 CPU, 64GB RAM, 16 GPU)  
**Framework**: PyTorch 2.6.0 with MPS + SDPA  
**Dataset**: GUI-Actor 1M samples via HuggingFace
