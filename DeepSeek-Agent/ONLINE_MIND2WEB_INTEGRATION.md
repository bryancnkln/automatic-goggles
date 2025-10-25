# Online-Mind2Web Dataset Integration for DeepSeek-Agent

This guide shows how to use the Online-Mind2Web benchmark dataset with DeepSeek-Agent for training and evaluation on macOS with MPS.

- Project: `https://github.com/OSU-NLP-Group/Online-Mind2Web`
- Dataset: `https://huggingface.co/datasets/osunlp/Online-Mind2Web`
- Paper: `https://arxiv.org/abs/2504.01382`

## 1) Download the dataset

```bash
pip install huggingface-hub

python << 'PY'
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="osunlp/Online-Mind2Web",
  repo_type="dataset",
  local_dir="data/online-mind2web"
)
print("✅ Online-Mind2Web downloaded to data/online-mind2web")
PY
```

## 2) Convert to training manifest

Use the provided converter to flatten trajectories into per-step samples:

```bash
python DeepSeek-Agent/scripts/prepare_online_mind2web_data.py \
  --dataset_dir data/online-mind2web \
  --output_manifest DeepSeek-Agent/logs/online_mind2web_manifest.jsonl
```

Output fields per line:
- `screenshot_path` (absolute path)
- `action_type`, `target_element`, `element_text`, `bbox`
- `instruction`, `task_context`, `success`
- `metadata.website`, `metadata.step_index`

## 3) Train on Online-Mind2Web

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python DeepSeek-Agent/src/training/train_projector.py \
  --manifest DeepSeek-Agent/logs/online_mind2web_manifest.jsonl \
  --model_dir DeepSeek-Agent/checkpoints/projector_online_mind2web \
  --epochs 2 \
  --batch_size 64 \
  --num_workers 16 \
  --device mps \
  --dtype float16
```

(Optional) Fine-tune with LoRA:
```bash
python DeepSeek-Agent/src/training/finetune_agent.py \
  --manifest DeepSeek-Agent/logs/online_mind2web_manifest.jsonl \
  --projector_path DeepSeek-Agent/checkpoints/projector_online_mind2web/model.pt \
  --lora_dir DeepSeek-Agent/checkpoints/agent_lora_online_mind2web \
  --batch_size 32 \
  --num_workers 16 \
  --device mps
```

## 4) Evaluation notes (WebJudge)

Online-Mind2Web proposes an LLM-as-a-judge evaluator (WebJudge) and emphasizes fair-use rules:
- Start from the specified websites (not general search)
- Include only factual actions in action history (no generated outputs)
- WebJudge with `o4-mini` achieves high human agreement; `WebJudge-7B` is also provided

See their README/paper for details:
- Repo: `https://github.com/OSU-NLP-Group/Online-Mind2Web`
- Paper: `https://arxiv.org/abs/2504.01382`

## 5) Tips for M4 Pro
- Use batch_size 48–64 for projector; 32 for LoRA
- Ensure `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- Close heavy apps to avoid memory contention

## 6) Troubleshooting
- If screenshots are missing, pass `--images_base` to the converter
- If entries appear without steps, verify the dataset split structure
- For slow IO, place `data/online-mind2web` on fast SSD
