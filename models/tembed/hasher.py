"""Learned Hash Embedder Training Module.

This module implements training for a neural network that maps float embeddings (1024-dim)
to binary hash codes (2048-dim bits) using contrastive learning.

Training Approach:
- Uses triplet loss with (anchor, positive, negative) samples
- Positives are similar embeddings (from Faiss top-k retrieval)
- Negatives are previous sample's anchor (hard negative mining)
- Loss preserves similarity: similar embeddings -> similar hash codes

Architecture:
- Input: 1024-dim float32 embedding
- Hidden layers with batch normalization and dropout
- Output: 2048-dim hash codes (via tanh -> sign for binary)

Usage:
    # Training
    python -m models.tembed.hasher -m train -ep 50

    # Inference test
    python -m models.tembed.hasher -m test
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tclogger import logger, TCLogbar, logstr, dict_to_str
from typing import Iterator, Optional

# Constants
PARENT_DIR = Path(__file__).parent
WEIGHTS_DIR = PARENT_DIR / "weights"
SAMPLES_DIR = Path("/media/data/tembed/train_samples")

# Model architecture
EMB_DIM = 1024
HASH_BITS = 2048
HIDDEN_DIM = 2048
DROPOUT = 0.1

# Training hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50

# Loss parameters
MARGIN = 0.5
QUANT_WEIGHT = 0.1
BALANCE_WEIGHT = 0.01


class EmbToHashNet(nn.Module):
    """Neural network for converting float embeddings to binary hash codes.

    Architecture:
        Input (1024) -> FC(2048) -> BN -> ReLU -> Dropout
                     -> FC(2048) -> BN -> ReLU -> Dropout
                     -> FC(2048) -> Tanh -> Hash bits

    The network is trained with contrastive loss to preserve similarity.
    During inference, the tanh output is binarized with sign function.
    """

    def __init__(
        self,
        input_dim: int = EMB_DIM,
        hash_bits: int = HASH_BITS,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hash_bits = hash_bits
        self.hidden_dim = hidden_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Hash projection layer
        self.hash_layer = nn.Linear(hidden_dim, hash_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning continuous hash codes.

        Args:
            x: Input embeddings with shape (batch, input_dim)

        Returns:
            Continuous hash codes with shape (batch, hash_bits), range [-1, 1]
        """
        h = self.encoder(x)
        h = self.hash_layer(h)
        h = torch.tanh(h)
        return h

    def get_hash_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Get binary hash codes (0/1) from input embeddings.

        Args:
            x: Input embeddings with shape (batch, input_dim)

        Returns:
            Binary hash codes with shape (batch, hash_bits), values 0 or 1
        """
        with torch.no_grad():
            h = self.forward(x)
            # Convert tanh output to binary: >0 -> 1, <=0 -> 0
            return (h > 0).float()


class TripletHashLoss(nn.Module):
    """Triplet loss for hash code learning.

    Combines:
    1. Triplet margin loss: anchor closer to positive than negative
    2. Quantization loss: encourage outputs to be close to -1 or 1
    3. Bit balance loss: encourage equal distribution of 0s and 1s

    The loss preserves similarity from original embeddings in hash space.
    """

    def __init__(
        self,
        margin: float = MARGIN,
        quant_weight: float = QUANT_WEIGHT,
        balance_weight: float = BALANCE_WEIGHT,
    ):
        super().__init__()
        self.margin = margin
        self.quant_weight = quant_weight
        self.balance_weight = balance_weight

    def hamming_distance(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """Compute differentiable Hamming distance using continuous codes.

        For continuous codes in [-1, 1], Hamming distance approximates as:
        d(h1, h2) = (hash_bits - h1 @ h2.T) / 2

        Args:
            h1, h2: Hash codes with shape (batch, hash_bits)

        Returns:
            Hamming distances with shape (batch,)
        """
        # Normalize to compute cosine similarity, then convert to distance
        # For tanh outputs in [-1,1]: dot product gives similarity
        dot_product = (h1 * h2).sum(dim=-1)
        # Convert to distance: higher dot product -> lower distance
        # Scale to [0, hash_bits] range
        distance = (h1.shape[-1] - dot_product) / 2
        return distance

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute triplet hash loss.

        Args:
            anchor: Anchor hash codes (batch, hash_bits)
            positive: Positive hash codes (batch, hash_bits)
            negative: Negative hash codes (batch, hash_bits)

        Returns:
            Total loss scalar and dict of individual loss components
        """
        # 1. Triplet margin loss
        d_ap = self.hamming_distance(anchor, positive)
        d_an = self.hamming_distance(anchor, negative)
        triplet_loss = F.relu(d_ap - d_an + self.margin).mean()

        # 2. Quantization loss: encourage values to be close to -1 or 1
        # |h|^2 should be close to 1 for each element
        quant_loss = (1 - anchor.pow(2)).mean()
        quant_loss += (1 - positive.pow(2)).mean()
        quant_loss += (1 - negative.pow(2)).mean()
        quant_loss /= 3

        # 3. Bit balance loss: mean of each bit should be close to 0
        # This encourages equal distribution of -1 and 1
        all_codes = torch.cat([anchor, positive, negative], dim=0)
        balance_loss = all_codes.mean(dim=0).pow(2).mean()

        # Combine losses
        total_loss = (
            triplet_loss
            + self.quant_weight * quant_loss
            + self.balance_weight * balance_loss
        )

        loss_dict = {
            "triplet": triplet_loss.item(),
            "quant": quant_loss.item(),
            "balance": balance_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict


class TrainSamplesDataset(Dataset):
    """PyTorch Dataset wrapper for TrainSamplesManager.

    Loads training samples from shards and provides (anchor, positive, negative)
    triplets for contrastive learning.

    Index Convention:
    - Both PyTorch Dataset and TrainSamplesManager now use 0-based indexing
    - Direct pass-through: dataset[idx] -> manager[idx]
    - All samples are valid, including sample 0 (uses random negative)

    Positive Sampling:
    - Each sample may have multiple positives (num_positives)
    - This class randomly selects one positive per training iteration
    - Different positives are selected across epochs for variety
    """

    def __init__(
        self,
        data_dir: str | Path,
        num_positives_to_use: int = 1,
        max_samples: int = None,
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing training samples
            num_positives_to_use: Number of positives to sample per anchor (default 1)
            max_samples: Maximum number of samples to use (default None for all)
        """
        # Import here to avoid circular imports
        from models.tembed.train import TrainSamplesManager

        self.manager = TrainSamplesManager(data_dir=data_dir, mode="read")
        self.num_positives_to_use = num_positives_to_use
        self._length = len(self.manager)
        if max_samples is not None:
            self._length = min(self._length, max_samples)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        """Get a training sample.

        Index Pass-through:
        - Input idx is 0-based (PyTorch Dataset convention)
        - Directly passed to TrainSamplesManager (now also 0-based)
        - manager[idx] handles sample 0 gracefully with random negative

        Random Positive Selection:
        - Randomly picks one positive from available positives
        - Provides variety across training iterations
        - Same anchor may be paired with different positives

        Args:
            idx: Sample index (0-based, directly maps to manager index)

        Returns:
            Dict with anchor_emb, pos_emb, neg_emb as numpy arrays
        """
        # Direct pass-through: both use 0-based indexing
        sample = self.manager[idx]

        # Randomly select one positive if multiple available
        pos_idx = np.random.randint(0, len(sample["pos_embs"]))

        return {
            "anchor_emb": sample["anchor_emb"].astype(np.float32),
            "pos_emb": sample["pos_embs"][pos_idx].astype(np.float32),
            "neg_emb": sample["neg_emb"].astype(np.float32),
        }


class HasherTrainer:
    """Trainer for the embedding-to-hash neural network.

    Handles:
    - Data loading from TrainSamplesManager
    - Model training with triplet loss
    - Checkpoint saving and loading
    - Training progress logging
    """

    def __init__(
        self,
        data_dir: str | Path = SAMPLES_DIR,
        input_dim: int = EMB_DIM,
        hash_bits: int = HASH_BITS,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        max_samples: int = None,
    ):
        """Initialize trainer.

        Args:
            data_dir: Directory containing training samples
            input_dim: Input embedding dimension (default 1024)
            hash_bits: Number of hash bits to generate (default 2048)
            hidden_dim: Hidden layer dimension (default 2048)
            dropout: Dropout rate (default 0.1)
            batch_size: Training batch size
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            max_samples: Maximum number of samples to use (default None for all)
        """
        self.data_dir = Path(data_dir)
        self.input_dim = input_dim
        self.hash_bits = hash_bits
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_samples = max_samples

        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[EmbToHashNet] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[TripletHashLoss] = None
        self.dataloader: Optional[DataLoader] = None

        # Paths for saving (will be set after init_model)
        self.weights_dir = WEIGHTS_DIR
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path: Optional[Path] = None
        self.best_path: Optional[Path] = None
        self.config_path: Optional[Path] = None

    def init_model(self):
        """Initialize the model and optimizer."""
        logger.note(f"> Initializing model:")
        info_dict = {
            "input_dim": self.input_dim,
            "hash_bits": self.hash_bits,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "device": str(self.device),
        }
        logger.mesg(dict_to_str(info_dict), indent=2)

        self.model = EmbToHashNet(
            input_dim=self.input_dim,
            hash_bits=self.hash_bits,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.criterion = TripletHashLoss(
            margin=MARGIN,
            quant_weight=QUANT_WEIGHT,
            balance_weight=BALANCE_WEIGHT,
        )

        # Generate checkpoint name from parameters
        name_parts = []
        name_parts.append(f"hb{self.hash_bits}")  # hash_bits
        name_parts.append(f"hd{self.hidden_dim}")  # hidden_dim
        name_parts.append(f"bs{self.batch_size}")  # batch_size
        name_parts.append(
            f"lr{self.learning_rate:.0e}".replace("e-0", "e-")
        )  # learning_rate
        if self.max_samples is not None:
            # Format max_samples in scientific notation
            if self.max_samples >= 1e6:
                name_parts.append(f"ms{self.max_samples / 1e6:.0f}m")
            elif self.max_samples >= 1e3:
                name_parts.append(f"ms{self.max_samples / 1e3:.0f}k")
            else:
                name_parts.append(f"ms{self.max_samples}")

        ckpt_name = "_".join(name_parts)
        hasher_ckpt = f"hasher_{ckpt_name}"
        self.ckpt_path = self.weights_dir / f"{hasher_ckpt}.pt"
        self.best_path = self.weights_dir / f"{hasher_ckpt}_best.pt"
        self.config_path = self.weights_dir / f"{hasher_ckpt}_config.json"

        # Save config
        config = {
            "input_dim": self.input_dim,
            "hash_bits": self.hash_bits,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_samples": self.max_samples,
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.mesg(f"  * checkpoint: {logstr.file(hasher_ckpt)}")

    def init_dataloader(self, num_workers: int = 4):
        """Initialize the data loader."""
        dataset = TrainSamplesDataset(self.data_dir, max_samples=self.max_samples)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        logger.note(f"> Loaded training dataset:")
        info_dict = {
            "data_dir": str(self.data_dir),
            "batch_size": self.batch_size,
            "dataset_len": len(dataset),
            "batches": len(self.dataloader),
        }
        logger.mesg(dict_to_str(info_dict), indent=2)

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save training checkpoint."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }
        if self.scheduler:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(ckpt, self.ckpt_path)
        if is_best:
            torch.save(ckpt, self.best_path)

    def load_checkpoint(self, ckpt_path: Path = None) -> int:
        """Load training checkpoint.

        Returns:
            Starting epoch number
        """
        if ckpt_path is None:
            ckpt_path = self.ckpt_path

        if not ckpt_path.exists():
            logger.warn(f"- Not found checkpoint:")
            logger.file(f"  * {ckpt_path}")
            return 0

        logger.note(f"> Loading checkpoint from:")
        logger.file(f"  * {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        logger.mesg(f"  epoch: {ckpt['epoch']}, loss: {ckpt['loss']:.4f}")
        return ckpt["epoch"]

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch.

        Returns:
            Dict with average losses for the epoch
        """
        self.model.train()

        total_losses = {"triplet": 0, "quant": 0, "balance": 0, "total": 0}
        num_batches = len(self.dataloader)

        bar = TCLogbar(total=num_batches)
        epoch_str = logstr.file(f"{epoch:03d}")
        bar.set_head(logstr.mesg(f"* [Epoch {epoch_str}/{self.epochs}]"))

        for batch_idx, batch in enumerate(self.dataloader):
            # Move data to device
            anchor = batch["anchor_emb"].to(self.device)
            positive = batch["pos_emb"].to(self.device)
            negative = batch["neg_emb"].to(self.device)
            # Forward pass
            anchor_hash = self.model(anchor)
            positive_hash = self.model(positive)
            negative_hash = self.model(negative)
            # Compute loss
            loss, loss_dict = self.criterion(anchor_hash, positive_hash, negative_hash)
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            # Accumulate losses
            for k, v in loss_dict.items():
                total_losses[k] += v
            # Update progress
            desc = (
                f"loss={loss_dict['total']:.4f} "
                f"(t={loss_dict['triplet']:.3f}, "
                f"q={loss_dict['quant']:.3f}, "
                f"b={loss_dict['balance']:.3f})"
            )
            bar.update(1, desc=desc)
        bar.update(flush=True, linebreak=True)

        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses

    def train(self, epochs: int = 20, resume: bool = True):
        """Train the model.

        Args:
            epochs: Number of epochs to train
            resume: Whether to resume from checkpoint
        """
        logger.note(f"> Starting training for {epochs} epochs")

        self.epochs = epochs
        start_epoch = 0
        if resume:
            start_epoch = self.load_checkpoint()

        best_loss = float("inf")

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs - start_epoch, eta_min=1e-6
        )

        for epoch in range(start_epoch + 1, epochs + 1):
            avg_losses = self.train_epoch(epoch)
            lr = self.optimizer.param_groups[0]["lr"]
            # Update scheduler
            self.scheduler.step()
            # Save checkpoint
            is_best = avg_losses["total"] < best_loss
            if is_best:
                best_loss = avg_losses["total"]
            self.save_checkpoint(epoch, avg_losses["total"], is_best=is_best)

        logger.success(f"> Training completed. Best loss: {best_loss:.4f}")


class HasherInference:
    """Inference wrapper for the trained hash model.

    Provides:
    - Model loading from checkpoint
    - Batch embedding to hash conversion
    - Hash code to hex string conversion
    """

    def __init__(
        self,
        weights_path: Path = None,
        config_path: Path = None,
        device: str = None,
    ):
        """Initialize inference.

        Args:
            weights_path: Path to model weights (default: auto-detect latest)
            config_path: Path to model config (default: auto-detect from weights)
            device: Device for inference (default: auto-detect)
        """
        self.weights_dir = WEIGHTS_DIR

        # Auto-detect weights if not provided
        if weights_path is None:
            weights_path = self._find_best_latest_weights()

        self.weights_path = weights_path

        # Auto-detect config path from weights path
        if config_path is None:
            # Replace .pt or _best.pt with _config.json
            config_name = self.weights_path.stem.replace("_best", "") + "_config.json"
            self.config_path = self.weights_dir / config_name
        else:
            self.config_path = config_path

        # Auto-detect device if not provided
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[EmbToHashNet] = None
        self._load_model()

    def _find_best_latest_weights(self) -> Path:
        """Find best and latest trained model weights.

        Priority:
        1. best: *_best.pt
        2. latest: *.pt file (sorted by change time)
        3. raise FileNotFoundError if not found

        Returns:
            Path to model weights
        """
        # Look for *_best.pt files
        best_files = list(self.weights_dir.glob("hasher_*_best.pt"))
        if best_files:
            # Return the most recently modified
            latest = max(best_files, key=lambda p: p.stat().st_mtime)
            return latest

        # Look for any hasher_*.pt files (checkpoints)
        ckpt_files = list(self.weights_dir.glob("hasher_*.pt"))
        if ckpt_files:
            # Filter out _best.pt files (already checked)
            ckpt_files = [f for f in ckpt_files if not f.stem.endswith("_best")]
            if ckpt_files:
                latest = max(ckpt_files, key=lambda p: p.stat().st_mtime)
                return latest

        raise FileNotFoundError(
            "No trained hasher model weights found in weights directory."
        )

    def _load_model(self):
        """Load model from checkpoint."""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

        # Load config
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config = json.load(f)
        else:
            config = {
                "input_dim": EMB_DIM,
                "hash_bits": HASH_BITS,
                "hidden_dim": HIDDEN_DIM,
                "dropout": 0.1,
            }

        # Initialize model
        self.model = EmbToHashNet(
            input_dim=config["input_dim"],
            hash_bits=config["hash_bits"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
        ).to(self.device)

        # Load weights
        ckpt = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        logger.note(f"> Loaded HasherInference model from:")
        logger.file(f"  * {self.weights_path}")

    def embed_to_hash(self, embeddings: np.ndarray) -> np.ndarray:
        """Convert float embeddings to binary hash codes.

        Args:
            embeddings: Float embeddings with shape (n, input_dim) or (input_dim,)

        Returns:
            Binary hash codes with shape (n, hash_bits) or (hash_bits,), dtype uint8
        """
        # Handle single embedding
        squeeze = False
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            squeeze = True

        # Convert to tensor
        x = torch.from_numpy(embeddings.astype(np.float32)).to(self.device)

        # Get hash codes
        with torch.no_grad():
            hash_codes = self.model.get_hash_codes(x)
            hash_codes = hash_codes.cpu().numpy().astype(np.uint8)

        if squeeze:
            return hash_codes[0]
        return hash_codes

    def hash_to_hex(self, hash_codes: np.ndarray) -> str:
        """Convert binary hash codes to hex string.

        Args:
            hash_codes: Binary hash codes with shape (hash_bits,)

        Returns:
            Hex string representation
        """
        # Pack bits into bytes
        n_bytes = (len(hash_codes) + 7) // 8
        padded = np.zeros(n_bytes * 8, dtype=np.uint8)
        padded[: len(hash_codes)] = hash_codes
        bytes_arr = np.packbits(padded)
        return bytes_arr.tobytes().hex()

    def embed_to_hex(self, embeddings: np.ndarray) -> list[str]:
        """Convert float embeddings directly to hex hash strings.

        Args:
            embeddings: Float embeddings with shape (n, input_dim) or (input_dim,)

        Returns:
            List of hex hash strings
        """
        # Handle single embedding
        squeeze = False
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            squeeze = True

        hash_codes = self.embed_to_hash(embeddings)
        hex_strs = [self.hash_to_hex(h) for h in hash_codes]

        if squeeze:
            return hex_strs[0]
        return hex_strs


class HasherArgParser(argparse.ArgumentParser):
    """Argument parser for hasher training and testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mode: train or test
        self.add_argument(
            "-m",
            "--mode",
            type=str,
            required=True,
            choices=["train", "test"],
            help="Operation mode",
        )
        # Training options
        self.add_argument(
            "-ep", "--epochs", type=int, default=EPOCHS, help="Number of epochs"
        )
        self.add_argument(
            "-bz", "--batch-size", type=int, default=BATCH_SIZE, help="Batch size"
        )
        self.add_argument(
            "-lr",
            "--learning-rate",
            type=float,
            default=LEARNING_RATE,
            help="Learning rate",
        )
        self.add_argument(
            "-w",
            "--overwrite",
            action="store_true",
            help="Overwrite existing checkpoint (train from scratch)",
        )
        # Model options
        self.add_argument(
            "-hb",
            "--hash-bits",
            type=int,
            default=HASH_BITS,
            help="Number of hash bits",
        )
        self.add_argument(
            "-hd",
            "--hidden-dim",
            type=int,
            default=HIDDEN_DIM,
            help="Hidden layer dimension",
        )
        self.add_argument(
            "-dp",
            "--dropout",
            type=float,
            default=DROPOUT,
            help="Dropout rate",
        )
        self.add_argument(
            "-ms",
            "--max-samples",
            type=int,
            default=None,
            help="Maximum number of training samples",
        )
        self.args, _ = self.parse_known_args()


def main():
    args = HasherArgParser().args

    if args.mode == "train":
        trainer = HasherTrainer(
            data_dir=SAMPLES_DIR,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hash_bits=args.hash_bits,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            max_samples=args.max_samples,
        )
        trainer.init_model()
        trainer.init_dataloader()
        trainer.train(epochs=args.epochs, resume=not args.overwrite)

    elif args.mode == "test":
        logger.note("> Testing HasherInference")
        inference = HasherInference()
        test_embs = np.random.randn(3, EMB_DIM).astype(np.float32)
        hash_codes = inference.embed_to_hash(test_embs)
        info_dict = {
            "input_shape": test_embs.shape,
            "hash_shape": hash_codes.shape,
            "hash_dtype": str(hash_codes.dtype),
            "sample_bits": hash_codes[0][:20].tolist(),
        }
        logger.mesg(dict_to_str(info_dict), indent=2)

        hex_strs = inference.embed_to_hex(test_embs)
        logger.mesg(f"  * Hex strings: {len(hex_strs)}")
        for i, h in enumerate(hex_strs):
            logger.mesg(f"  - [{i}]: {h[:32]}... (len={len(h)})")


if __name__ == "__main__":
    main()
    # Case: mode train
    # python -m models.tembed.hasher -m train -hb 2048 -hd 1024 -ms 50000 -ep 10
    # python -m models.tembed.hasher -m train -hb 2048 -ms 1000000 -ep 50
    # python -m models.tembed.hasher -m train -ep 50 -dp 0.2
    # python -m models.tembed.hasher -m train -w

    # Case: mode test
    # python -m models.tembed.hasher -m test
