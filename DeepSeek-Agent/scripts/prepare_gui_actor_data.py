#!/usr/bin/env python3
"""
Convert GUI-Actor dataset format to DeepSeek-Agent training manifest.

Handles:
- Multiple GUI-Actor sources (UGround, GUIEnv, GUIAct, AMEX, AndroidControl, Wave-UI)
- JSON and JSONL formats
- Flexible path resolution
- Train/val splitting
"""

import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON or JSONL file."""
    entries = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                # JSONL format: one JSON object per line
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            else:
                # Standard JSON format: array of objects
                data = json.load(f)
                if isinstance(data, list):
                    entries = data
                else:
                    entries = [data]
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
    
    return entries


def resolve_image_path(entry: Dict[str, Any], images_base: str) -> str:
    """Resolve image path from GUI-Actor entry."""
    # Try multiple field names
    for field in ['screenshot_path', 'image_path', 'img_path', 'path']:
        if field in entry:
            img_rel = entry[field]
            full_path = os.path.join(images_base, img_rel)
            
            # Try direct path
            if os.path.exists(full_path):
                return full_path
            
            # Try parent directory resolution
            for root, dirs, files in os.walk(images_base):
                for file in files:
                    if file == os.path.basename(img_rel):
                        return os.path.join(root, file)
    
    return None


def convert_gui_actor_entry(
    entry: Dict[str, Any],
    images_base: str
) -> List[Dict[str, Any]]:
    """
    Convert single GUI-Actor entry to training manifest format.
    
    Returns list of training entries (one per annotation).
    """
    training_entries = []
    
    # Resolve image path
    img_path = resolve_image_path(entry, images_base)
    if not img_path:
        logger.warning(f"Could not find image for entry {entry.get('image_id', 'unknown')}")
        return []
    
    # Extract common fields
    image_id = entry.get('image_id', '')
    task_context = entry.get('task_context', '')
    success = entry.get('success', True)
    width = entry.get('width', 1920)
    height = entry.get('height', 1080)
    
    metadata = entry.get('metadata', {})
    if not isinstance(metadata, dict):
        metadata = {}
    
    # Extract source info
    source = metadata.get('source', entry.get('source', 'gui-actor'))
    app = metadata.get('app', entry.get('app', 'unknown'))
    
    # Process annotations
    annotations = entry.get('annotations', [])
    if not annotations:
        # Create single entry with no specific annotation
        training_entries.append({
            'screenshot_path': img_path,
            'image_id': image_id,
            'action_type': 'none',
            'target_element': '',
            'instruction': task_context,
            'bbox': [],
            'element_text': '',
            'task_context': task_context,
            'success': success,
            'metadata': {
                'source': source,
                'app': app,
                'resolution': f"{width}x{height}",
                'has_annotations': False
            }
        })
    else:
        # Create entry for each annotation
        for ann in annotations:
            # Validate annotation
            if not isinstance(ann, dict):
                continue
            
            action_type = ann.get('action_type', 'click')
            target_element = ann.get('target_element', '')
            bbox = ann.get('bbox', [])
            element_text = ann.get('element_text', '')
            
            # Create instruction from annotation or use task context
            if 'instruction' in ann:
                instruction = ann['instruction']
            elif target_element:
                instruction = f"{action_type} {target_element}. {task_context}".strip()
            else:
                instruction = task_context
            
            training_entries.append({
                'screenshot_path': img_path,
                'image_id': image_id,
                'action_type': action_type,
                'target_element': target_element,
                'instruction': instruction,
                'bbox': bbox,
                'element_text': element_text,
                'task_context': task_context,
                'success': success and ann.get('success', True),
                'metadata': {
                    'source': source,
                    'app': app,
                    'resolution': f"{width}x{height}",
                    'has_annotations': True
                }
            })
    
    return training_entries


def convert_gui_actor_to_manifest(
    input_json_files: List[str],
    images_base: str,
    output_manifest: str,
) -> Dict[str, int]:
    """
    Convert GUI-Actor JSON files to training manifest.
    
    Returns statistics dictionary.
    """
    manifest_entries = []
    stats = {
        'total_processed': 0,
        'valid_entries': 0,
        'failed_entries': 0,
        'total_annotations': 0
    }
    
    # Process each JSON file
    for json_file in input_json_files:
        logger.info(f"Processing {json_file}...")
        
        # Load entries
        entries = load_json_file(json_file)
        if not entries:
            logger.warning(f"No entries found in {json_file}")
            continue
        
        # Convert each entry
        for entry in tqdm(entries, desc=f"Converting {os.path.basename(json_file)}"):
            stats['total_processed'] += 1
            
            try:
                training_entries = convert_gui_actor_entry(entry, images_base)
                if training_entries:
                    manifest_entries.extend(training_entries)
                    stats['valid_entries'] += 1
                    stats['total_annotations'] += len(training_entries)
                else:
                    stats['failed_entries'] += 1
            except Exception as e:
                logger.warning(f"Error converting entry: {e}")
                stats['failed_entries'] += 1
    
    # Write manifest
    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    with open(output_manifest, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    stats['output_path'] = output_manifest
    stats['output_entries'] = len(manifest_entries)
    
    return stats


def print_stats(stats: Dict[str, Any]) -> None:
    """Print conversion statistics."""
    print("\n" + "="*60)
    print("CONVERSION STATISTICS")
    print("="*60)
    print(f"Total processed entries:    {stats['total_processed']}")
    print(f"Valid entries:              {stats['valid_entries']}")
    print(f"Failed entries:             {stats['failed_entries']}")
    print(f"Total annotations created:  {stats['total_annotations']}")
    print(f"Output manifest entries:    {stats['output_entries']}")
    print(f"Output path:                {stats['output_path']}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert GUI-Actor dataset to DeepSeek-Agent training manifest'
    )
    parser.add_argument(
        '--input_json',
        nargs='+',
        required=True,
        help='GUI-Actor JSON files (supports glob patterns)'
    )
    parser.add_argument(
        '--images_base',
        required=True,
        help='Base directory for image resolution'
    )
    parser.add_argument(
        '--output_manifest',
        required=True,
        help='Output JSONL manifest path'
    )
    
    args = parser.parse_args()
    
    # Expand glob patterns
    json_files = []
    for pattern in args.input_json:
        matched = glob.glob(pattern)
        if matched:
            json_files.extend(matched)
        else:
            # Try as direct path
            if os.path.exists(pattern):
                json_files.append(pattern)
            else:
                logger.warning(f"No files matched pattern: {pattern}")
    
    if not json_files:
        logger.error("No JSON files found!")
        return
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    for f in json_files:
        logger.info(f"  - {f}")
    
    # Convert
    stats = convert_gui_actor_to_manifest(
        json_files,
        args.images_base,
        args.output_manifest
    )
    
    # Print statistics
    print_stats(stats)
    
    # Success/failure summary
    if stats['failed_entries'] == 0:
        logger.info("✅ Conversion completed successfully!")
    else:
        logger.warning(
            f"⚠️  Conversion completed with {stats['failed_entries']} failures. "
            "Check logs above."
        )


if __name__ == '__main__':
    main()
