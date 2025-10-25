"""
Stage 1: Train Vision Projector

Trains the 2-layer MLP (2048 → 4096) to align vision tokens with LLM space.
"""

import logging
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

from ..vision.vision_projector import VisionProjector

logger = logging.getLogger(__name__)


class VisionProjectorDataset(Dataset):
    """Dataset for projector training."""
    
    def __init__(self, manifest_path: str):
        """
        Initialize dataset.
        
        Args:
            manifest_path: Path to JSONL file with {vision_tokens_path, target_embedding_path}
        """
        self.samples = []
        
        with open(manifest_path) as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load vision tokens
        vision_tokens = np.load(sample["vision_tokens_path"]).astype(np.float32)  # (N, 2048)
        
        # Load target embeddings
        target_emb = np.load(sample["target_embedding_path"]).astype(np.float32)  # (N, 4096)
        
        return (
            torch.from_numpy(vision_tokens),
            torch.from_numpy(target_emb),
        )


def train_projector(
    projector: VisionProjector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    num_epochs: int,
    learning_rate: float,
    warmup_steps: int,
    save_dir: str,
):
    """
    Train the vision projector.
    
    Args:
        projector: VisionProjector model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Number of epochs
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        save_dir: Directory to save checkpoints
    """
    projector = projector.to(device)
    
    optimizer = optim.Adam(projector.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Linear warmup scheduler
    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * (step / warmup_steps)
        return learning_rate
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        # Training
        projector.train()
        train_loss = 0.0
        
        for batch_idx, (vision_tokens, target_emb) in enumerate(train_loader):
            vision_tokens = vision_tokens.to(device)  # (B, N, 2048)
            target_emb = target_emb.to(device)         # (B, N, 4096)
            
            # Forward
            projected = projector(vision_tokens)  # (B, N, 4096)
            
            # Loss
            loss = criterion(projected, target_emb)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            global_step += 1
            
            if batch_idx % 100 == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch} Batch {batch_idx} Loss: {loss:.4f} LR: {lr:.2e}"
                )
        
        # Validation
        projector.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for vision_tokens, target_emb in val_loader:
                vision_tokens = vision_tokens.to(device)
                target_emb = target_emb.to(device)
                
                projected = projector(vision_tokens)
                loss = criterion(projected, target_emb)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        logger.info(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state": projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            }
            torch.save(checkpoint, save_path / "best_model.pt")
            logger.info(f"Saved best model with val_loss {val_loss:.4f}")
        
        # Save epoch checkpoint
        torch.save(projector.state_dict(), save_path / f"epoch_{epoch}.pt")
    
    logger.info(f"Training complete. Best model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train vision projector")
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest")
    parser.add_argument("--save_dir", default="checkpoints/projector_stage_a")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create dataset
    dataset = VisionProjectorDataset(args.manifest)
    
    # Split into train/val
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Create projector
    projector = VisionProjector(
        in_dim=2048,
        out_dim=4096,
    )
    
    logger.info(f"Projector params: {projector.get_num_params():,}")
    
    # Train
    train_projector(
        projector=projector,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
