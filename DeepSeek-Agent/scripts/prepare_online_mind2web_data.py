#!/usr/bin/env python3
"""
Convert Online-Mind2Web dataset to DeepSeek-Agent training manifest (JSONL).

This script is intentionally tolerant to schema variations across versions.
It extracts per-step samples with screenshot + action metadata to a flat JSONL
manifest consumable by projector training and LoRA fine-tuning.

Input assumptions (best-effort, supports multiple styles):
- Root contains JSON/JSONL files describing tasks/trajectories, e.g.:
  - tasks.jsonl / trajectories.jsonl / *.json
- Each task has fields like: {task_id, instruction, description, steps/trajectory}
- Each step may include: screenshot(_path|_file), action {type, bbox, input_text, ...}
- Screenshots are stored beneath dataset_dir or a separate images dir

Usage:
  python prepare_online_mind2web_data.py \
    --dataset_dir data/online-mind2web \
    --output_manifest DeepSeek-Agent/logs/online_mind2web_manifest.jsonl \
    [--images_base data/online-mind2web]

Output JSONL fields:
- screenshot_path, image_id, action_type, target_element, instruction,
  bbox, element_text, task_context, success, metadata{source, website, step_index, resolution}
"""

import argparse
import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prepare_online_mind2web")


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                return [json.loads(line) for line in f if line.strip()]
            data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]
    except Exception as e:
        logger.warning(f"Failed reading {path}: {e}")
        return []


def _resolve_image_path(img_rel: Optional[str], images_base: str, dataset_dir: str) -> Optional[str]:
    if not img_rel:
        return None
    candidates = [
        os.path.join(images_base, img_rel),
        os.path.join(dataset_dir, img_rel),
        img_rel,
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # last resort: search by basename
    basename = os.path.basename(img_rel)
    for root, _, files in os.walk(images_base or dataset_dir):
        if basename in files:
            return os.path.join(root, basename)
    return None


def _to_list(x):
    return x if isinstance(x, list) else ([] if x is None else [x])


def convert_online_mind2web(
    dataset_dir: str,
    output_manifest: str,
    images_base: Optional[str] = None,
) -> Dict[str, Any]:
    images_base = images_base or dataset_dir

    # Collect candidate json files
    patterns = [
        os.path.join(dataset_dir, "**", "*.jsonl"),
        os.path.join(dataset_dir, "**", "*.json"),
    ]
    json_files: List[str] = []
    for p in patterns:
        json_files.extend(glob.glob(p, recursive=True))

    # Heuristics: prefer files named trajectories/tasks
    preferred = [
        f for f in json_files
        if any(k in os.path.basename(f).lower() for k in ["trajectory", "trajector", "task", "online_mind2web"])
    ]
    if preferred:
        json_files = preferred

    if not json_files:
        raise FileNotFoundError("No JSON/JSONL files found in dataset_dir")

    stats = {
        "files": len(json_files),
        "entries": 0,
        "steps": 0,
        "written": 0,
        "missing_images": 0,
    }

    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    out = open(output_manifest, "w", encoding="utf-8")

    for path in json_files:
        records = _read_json_or_jsonl(path)
        for rec in tqdm(records, desc=f"{os.path.basename(path)}"):
            stats["entries"] += 1

            task_id = rec.get("task_id") or rec.get("id") or rec.get("uuid") or ""
            website = rec.get("website") or rec.get("domain") or rec.get("site") or ""
            instruction = rec.get("instruction") or rec.get("goal") or rec.get("query") or ""
            task_context = rec.get("description") or rec.get("context") or ""
            success = bool(rec.get("success", True))

            steps = rec.get("steps") or rec.get("trajectory") or rec.get("actions") or []
            steps = _to_list(steps)

            for idx, step in enumerate(steps):
                stats["steps"] += 1
                # image path variants
                img_rel = (
                    step.get("screenshot")
                    or step.get("screenshot_path")
                    or step.get("image")
                    or step.get("image_path")
                )
                img_path = _resolve_image_path(img_rel, images_base, dataset_dir)
                if not img_path:
                    stats["missing_images"] += 1
                    continue

                # action
                action = step.get("action") or step.get("act") or {}
                action_type = action.get("type") or action.get("name") or step.get("type") or "none"
                element_text = action.get("text") or action.get("value") or step.get("text") or ""
                bbox = (
                    action.get("bbox")
                    or step.get("bbox")
                    or action.get("target_bbox")
                    or []
                )
                target_element = action.get("target") or action.get("selector") or ""

                manifest_entry = {
                    "screenshot_path": img_path,
                    "image_id": f"{task_id}:{idx}",
                    "action_type": action_type,
                    "target_element": target_element,
                    "instruction": instruction,
                    "bbox": bbox,
                    "element_text": element_text,
                    "task_context": task_context,
                    "success": success and bool(step.get("success", True)),
                    "metadata": {
                        "source": "online-mind2web",
                        "website": website,
                        "step_index": idx,
                    },
                }
                out.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
                stats["written"] += 1

    out.close()
    logger.info(f"Wrote {stats['written']} entries to {output_manifest}")
    logger.info(f"Missing images: {stats['missing_images']}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_manifest", required=True)
    ap.add_argument("--images_base", default=None)
    args = ap.parse_args()

    stats = convert_online_mind2web(
        dataset_dir=args.dataset_dir,
        output_manifest=args.output_manifest,
        images_base=args.images_base,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
