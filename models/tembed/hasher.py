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
from tclogger import TCLogger, TCLogbar, logstr, dict_to_str, int_bits
from typing import Optional

logger = TCLogger("Hasher")

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
# MARGIN_RATIO is relative to HASH_BITS (e.g., 0.2 = 20% of hash bits)
# For HASH_BITS=2048, MARGIN_RATIO=0.2 means MARGIN=409.6
# This requires d(anchor, neg) - d(anchor, pos) > 20% of hash bits
MARGIN_RATIO = 0.2
# Reduce quantization and balance weights to prioritize triplet and similarity losses
QUANT_WEIGHT = 0.01
BALANCE_WEIGHT = 0.001
# CRITICAL FIX #1: SIM_WEIGHT must be large enough to prevent range shift
# Problem: MSE similarity loss has magnitude ~0.25, while triplet loss is ~400 (initially)
# With SIM_WEIGHT=0.2, sim contributes only 0.05, completely dominated by triplet
# This causes hash similarities to shift upward (+0.14 offset from embedding similarities)
# Solution: Set SIM_WEIGHT=1000 to make similarity preservation the PRIMARY objective
# At convergence: triplet~20, sim~0.2, weighted: 20 vs 200, sim dominates 10:1
SIM_WEIGHT = 1000.0

# CRITICAL FIX #2: Use Straight-Through Estimator (STE) during training
# Problem: Training uses continuous hash codes [-1,1], but inference uses binary {0,1}
# This mismatch causes the model to learn continuous optimizations that don't transfer
# to binary codes, resulting in poor benchmark performance (0.3-0.4 vs expected >0.6)
# Solution: Use STE to train with binary codes {-1,1} while maintaining gradients
# - Forward: h_binary = sign(tanh(x)), producing {-1, 1}
# - Backward: gradients flow through tanh (straight-through estimator)
# This ensures training behavior matches inference binary quantization


class EmbToHashNet(nn.Module):
    """Neural network for converting float embeddings to binary hash codes.

    Architecture:
        Input (1024) -> FC(2048) -> ReLU -> Dropout
                     -> FC(2048) -> ReLU -> Dropout
                     -> FC(2048) -> Tanh -> Hash bits

    IMPORTANT: BatchNorm Removal
    ----------------------------
    BatchNorm was removed from this architecture to fix a critical bug that caused
    variance collapse and made all embeddings produce nearly identical hash codes.

    The Problem:
        Original architecture: Linear -> BatchNorm -> ReLU -> Dropout
        - Linear outputs with mean≈0, variance≈1
        - BatchNorm normalizes to mean=0, variance=1
        - ReLU zeros negative values, reducing variance from 1 to ~0.25
        - During training, BN's running_var recorded extremely small values (~0.007)
        - During inference, division by sqrt(running_var) ≈ sqrt(0.007) ≈ 0.084
        - Result: All inputs compressed to nearly identical values!

    Evidence from broken model:
        - Different embeddings (zeros vs ones) had only 11 bits difference out of 2048 (<1%)
        - Expected: ~1024 bits difference (~50% for independent inputs)
        - BatchNorm running_var showed: 1998/2048 channels had variance < 0.01

    The Fix:
        Current architecture: Linear -> ReLU -> Dropout (no BatchNorm)
        - Simpler, more stable training
        - Proper input separation: ~40-50% Hamming distance for different inputs
        - Dropout provides sufficient regularization

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
        # Note: BN after activation causes issues. Use BN -> ReLU order or remove BN.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Hash projection layer
        self.hash_layer = nn.Linear(hidden_dim, hash_bits)

    def forward(self, x: torch.Tensor, use_ste: bool = False) -> torch.Tensor:
        """Forward pass returning continuous or binary hash codes.

        Args:
            x: Input embeddings with shape (batch, input_dim)
            use_ste: If True, use straight-through estimator for binary codes

        Returns:
            Hash codes with shape (batch, hash_bits)
            - If use_ste=False: continuous codes in range [-1, 1]
            - If use_ste=True: binary codes {-1, 1} with gradients from tanh
        """
        h = self.encoder(x)
        h = self.hash_layer(h)
        h = torch.tanh(h)

        if use_ste:
            # Straight-through estimator: forward uses binary {-1,1}, backward uses tanh gradient
            # This ensures training and inference use the same scale
            h_binary = torch.sign(h)  # Convert to {-1, 0, 1}
            # Handle the rare case of exactly 0 (treat as +1)
            h_binary = torch.where(h_binary == 0, torch.ones_like(h_binary), h_binary)
            # Straight-through: forward = binary, backward = continuous gradient
            h = h_binary + h - h.detach()

        return h

    def get_hash_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Get binary hash codes (0/1) from input embeddings.

        Args:
            x: Input embeddings with shape (batch, input_dim)

        Returns:
            Binary hash codes with shape (batch, hash_bits), values 0 or 1
        """
        with torch.no_grad():
            h = self.forward(x, use_ste=False)  # Always use continuous for inference
            # Convert tanh output to binary: >0 -> 1, <=0 -> 0
            return (h > 0).to(torch.uint8)


class TripletHashLoss(nn.Module):
    """Triplet loss for hash code learning.

    Combines:
    1. Triplet margin loss: anchor closer to positive than negative
    2. Quantization loss: encourage outputs to be close to -1 or 1
    3. Bit balance loss: encourage equal distribution of 0s and 1s
    4. Similarity preservation loss: preserve relative similarities from embedding space

    The loss preserves similarity from original embeddings in hash space.
    """

    def __init__(
        self,
        hash_bits: int = HASH_BITS,
        margin_ratio: float = MARGIN_RATIO,
        quant_weight: float = QUANT_WEIGHT,
        balance_weight: float = BALANCE_WEIGHT,
        sim_weight: float = SIM_WEIGHT,
    ):
        super().__init__()
        self.hash_bits = hash_bits
        self.margin = hash_bits * margin_ratio  # Convert ratio to absolute value
        self.quant_weight = quant_weight
        self.balance_weight = balance_weight
        self.sim_weight = sim_weight

    def hamming_distance(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """Compute differentiable Hamming distance for binary codes {-1, 1}.

        For binary codes in {-1, 1}, Hamming distance is the count of differing bits.
        When h1[i] != h2[i]: (h1[i] - h2[i])^2 = 4 (e.g., 1-(-1)=2, squared=4)
        When h1[i] == h2[i]: (h1[i] - h2[i])^2 = 0

        So: hamming_distance = sum((h1 - h2)^2) / 4

        Args:
            h1, h2: Binary hash codes with shape (batch, hash_bits), values in {-1, 1}

        Returns:
            Hamming distances with shape (batch,)
        """
        # For binary {-1,1}: (h1-h2)^2 is 4 when different, 0 when same
        distance = ((h1 - h2) ** 2).sum(dim=-1) / 4
        return distance

    def compute_quantization_loss(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """Compute quantization loss to encourage binary values {-1, 1}.

        For {-1,1} codes: loss = 1 - h^2
        This is 0 when h=-1 or h=1, maximum 1 when h=0
        """
        quant_loss = (1 - anchor.pow(2)).mean()
        quant_loss += (1 - positive.pow(2)).mean()
        quant_loss += (1 - negative.pow(2)).mean()
        return quant_loss / 3

    def compute_balance_loss(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """Compute bit balance loss: mean of each bit should be close to 0 for {-1,1}."""
        all_codes = torch.cat([anchor, positive, negative], dim=0)
        bit_means = all_codes.mean(dim=0)  # Mean of each bit across batch
        return bit_means.pow(2).mean()

    def compute_similarity_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
        d_ap: torch.Tensor,
        d_an: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity preservation loss.

        Encourages hash similarity to correlate with embedding similarity.
        """
        if anchor_emb is None or positive_emb is None or self.sim_weight == 0:
            return torch.tensor(0.0, device=anchor.device)

        # Normalize embeddings
        anchor_emb_norm = F.normalize(anchor_emb, p=2, dim=-1)
        positive_emb_norm = F.normalize(positive_emb, p=2, dim=-1)

        # Compute embedding cosine similarity (anchor-positive)
        emb_sim_ap = (anchor_emb_norm * positive_emb_norm).sum(dim=-1)

        # Compute hash similarity from Hamming distance
        hash_sim_ap = 1 - d_ap / anchor.shape[-1]

        # MSE loss for anchor-positive similarity preservation
        sim_loss_ap = F.mse_loss(hash_sim_ap, emb_sim_ap)

        # Also preserve dissimilarity for anchor-negative (if available)
        if negative_emb is not None:
            negative_emb_norm = F.normalize(negative_emb, p=2, dim=-1)
            emb_sim_an = (anchor_emb_norm * negative_emb_norm).sum(dim=-1)
            hash_sim_an = 1 - d_an / anchor.shape[-1]
            sim_loss_an = F.mse_loss(hash_sim_an, emb_sim_an)
            return (sim_loss_ap + sim_loss_an) / 2

        return sim_loss_ap

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        anchor_emb: torch.Tensor = None,
        positive_emb: torch.Tensor = None,
        negative_emb: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute triplet hash loss.

        Args:
            anchor: Anchor hash codes (batch, hash_bits)
            positive: Positive hash codes (batch, hash_bits)
            negative: Negative hash codes (batch, hash_bits)
            anchor_emb: Original anchor embeddings for similarity preservation
            positive_emb: Original positive embeddings for similarity preservation
            negative_emb: Original negative embeddings for similarity preservation

        Returns:
            Total loss scalar and dict of individual loss components
        """
        # 1. Triplet margin loss
        d_ap = self.hamming_distance(anchor, positive)
        d_an = self.hamming_distance(anchor, negative)
        triplet_loss = F.relu(d_ap - d_an + self.margin).mean()

        # 2. Quantization loss
        quant_loss = self.compute_quantization_loss(anchor, positive, negative)

        # 3. Bit balance loss
        balance_loss = self.compute_balance_loss(anchor, positive, negative)

        # 4. Similarity preservation loss
        sim_loss = self.compute_similarity_loss(
            anchor,
            positive,
            negative,
            anchor_emb,
            positive_emb,
            negative_emb,
            d_ap,
            d_an,
        )

        # Combine losses
        total_loss = (
            triplet_loss
            + self.quant_weight * quant_loss
            + self.balance_weight * balance_loss
            + self.sim_weight * sim_loss
        )

        loss_dict = {
            "triplet": triplet_loss.item(),
            "quant": quant_loss.item(),
            "balance": balance_loss.item(),
            "sim": sim_loss.item() if isinstance(sim_loss, torch.Tensor) else sim_loss,
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
        remove_weights: bool = False,
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
        self.remove_weights = remove_weights

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

    def _log_model_info(self):
        """Log model initialization information."""
        logger.note(f"> Initializing model:")
        info_dict = {
            "input_dim": self.input_dim,
            "hash_bits": self.hash_bits,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "device": str(self.device),
        }
        logger.mesg(dict_to_str(info_dict), indent=2)

    def _create_model(self):
        """Create and initialize the hash embedding model."""
        self.model = EmbToHashNet(
            input_dim=self.input_dim,
            hash_bits=self.hash_bits,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

    def _create_optimizer_and_criterion(self):
        """Create optimizer and loss criterion."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.criterion = TripletHashLoss(
            hash_bits=self.hash_bits,
            margin_ratio=MARGIN_RATIO,
            quant_weight=QUANT_WEIGHT,
            balance_weight=BALANCE_WEIGHT,
            sim_weight=0.2,  # Increase to prioritize similarity preservation
        )

    def _generate_checkpoint_paths(self):
        """Generate checkpoint file paths based on model parameters."""
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

        logger.mesg(f"  * checkpoint : {logstr.file(hasher_ckpt)}")

    def _handle_weight_removal(self):
        """Remove existing weight files if requested."""
        if not self.remove_weights:
            return

        weights_to_remove = [
            self.ckpt_path,
            self.best_path,
            self.config_path,
        ]
        removed_count = 0
        for weight_path in weights_to_remove:
            if weight_path.exists():
                weight_path.unlink()
                removed_count += 1
        if removed_count > 0:
            logger.warn(f"  × Removed {removed_count} existing weight file(s)")

    def _save_model_config(self):
        """Save model configuration to JSON file."""
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

    def init_model(self):
        """Initialize the model and optimizer."""
        self._log_model_info()

        self._create_model()
        self._create_optimizer_and_criterion()
        self._generate_checkpoint_paths()
        self._handle_weight_removal()
        self._save_model_config()

    def init_dataloader(self, num_workers: int = 0, preload_shards: bool = True):
        """Initialize the data loader.

        Note: num_workers=0 (single-process) is used to avoid multiprocessing issues
        with large datasets (1M+ samples). Each worker would initialize TrainSamplesManager,
        causing I/O contention and potential deadlocks. Single-process loading is actually
        efficient since data is memory-mapped from shards.

        Args:
            num_workers: Number of worker processes (default 0 for single-process)
            preload_shards: Whether to preload shards that cover max_samples (default True)
        """
        dataset = TrainSamplesDataset(self.data_dir, max_samples=self.max_samples)

        # Preload shards covering max_samples to avoid slow first batch
        if preload_shards and self.max_samples:
            num_shards_needed = (self.max_samples - 1) // 100000 + 1  # shard_size=100k
            logger.note(
                f"> Preloading {num_shards_needed} shards for {self.max_samples} samples..."
            )
            for shard_idx in range(num_shards_needed):
                dataset.manager.shards_reader.load_shard(shard_idx)
            logger.okay(f"  ✓ Shards preloaded")

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

    def load_checkpoint(self, ckpt_path: Path = None) -> tuple[int, float]:
        """Load training checkpoint.

        Returns:
            Tuple of (starting_epoch, best_loss)
        """
        if ckpt_path is None:
            ckpt_path = self.ckpt_path

        if not ckpt_path.exists():
            logger.warn(f"  * No existed checkpoint, training from scratch.")
            return 0, float("inf")

        logger.mesg(f"  > Loading checkpoint from:")
        logger.file(f"    * {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        logger.mesg(f"  epoch: {ckpt['epoch']}, loss: {ckpt['loss']:.4f}")
        return ckpt["epoch"], ckpt.get("loss", float("inf"))

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch.

        Returns:
            Dict with average losses for the epoch
        """
        self.model.train()

        total_losses = {"triplet": 0, "quant": 0, "balance": 0, "sim": 0, "total": 0}
        num_batches = len(self.dataloader)

        bar = TCLogbar(total=num_batches)
        epoch_str = logstr.file(f"{epoch:>{int_bits(self.epochs)}d}")
        bar.set_head(logstr.mesg(f"  * [Epoch {epoch_str}/{self.epochs}]"))

        for batch_idx, batch in enumerate(self.dataloader):
            # Move data to device
            anchor_emb = batch["anchor_emb"].to(self.device)
            positive_emb = batch["pos_emb"].to(self.device)
            negative_emb = batch["neg_emb"].to(self.device)
            # Forward pass with STE for binary codes
            anchor_hash = self.model(anchor_emb, use_ste=True)
            positive_hash = self.model(positive_emb, use_ste=True)
            negative_hash = self.model(negative_emb, use_ste=True)
            # Compute loss with original embeddings for similarity preservation
            loss, loss_dict = self.criterion(
                anchor_hash,
                positive_hash,
                negative_hash,
                anchor_emb=anchor_emb,
                positive_emb=positive_emb,
                negative_emb=negative_emb,
            )
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
                f"b={loss_dict['balance']:.3f}, "
                f"s={loss_dict['sim']:.3f})"
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
        logger.note(f"> Training for {epochs} epochs:")

        self.epochs = epochs
        start_epoch = 0
        best_loss = float("inf")

        if resume:
            start_epoch, best_loss = self.load_checkpoint()

        # Check if training is already completed
        if start_epoch >= epochs:
            logger.okay(
                f"> Training already completed at epoch {start_epoch}. Best loss: {best_loss:.4f}"
            )
            return

        # Warmup: preload shards by fetching first batch
        # This prevents hanging during first epoch due to lazy shard loading
        if start_epoch == 0:
            logger.note("> Warming up data loader (preloading shards)...")
            try:
                warmup_batch = next(iter(self.dataloader))
                logger.okay(f"  ✓ First batch loaded successfully")
            except Exception as e:
                logger.warn(f"  ✗ Warmup failed (will retry during training): {e}")

        # Learning rate scheduler
        remaining_epochs = epochs - start_epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=remaining_epochs, eta_min=1e-6
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

        logger.okay(f"> Training completed. Best loss: {best_loss:.4f}")


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

    def emb_to_hash(self, embs: np.ndarray) -> np.ndarray:
        """Convert float embeddings to binary hash codes.

        Args:
            embs: Float embeddings with shape (n, input_dim) or (input_dim,)

        Returns:
            Binary hash codes with shape (n, hash_bits) or (hash_bits,), dtype uint8
        """
        squeeze = False
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
            squeeze = True
        x = torch.from_numpy(embs.astype(np.float32)).to(self.device)
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

    def emb_to_hex(self, embs: np.ndarray) -> list[str]:
        """Convert float embeddings directly to hex hash strings.

        Args:
            embs: Float embeddings with shape (n, input_dim) or (input_dim,)
        Returns:
            List of hex hash strings
        """
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        hash_codes = self.emb_to_hash(embs)
        hex_strs = [self.hash_to_hex(h) for h in hash_codes]
        return hex_strs


class HasherInferenceTester:
    """Tester for HasherInference model validation.

    Performs comprehensive tests to verify model produces diverse hash codes:
    - Test 1: Random embeddings with normal distribution
    - Test 2: Batch vs individual processing consistency
    - Test 3: Extreme value separation (zeros, ones, random)

    A healthy model should produce:
    - ~40-50% Hamming distance for independent random inputs
    - Consistent results between batch and individual processing
    - Strong separation for very different inputs (zeros vs ones)
    """

    def __init__(self, inference: Optional[HasherInference] = None):
        """Initialize tester.

        Args:
            inference: HasherInference instance. If None, creates a new one.
        """
        self.inference = inference or HasherInference()

    def test_random_embeddings(self, seed: int = 42, n_samples: int = 3):
        """Test with random embeddings from normal distribution.

        Args:
            seed: Random seed for reproducibility
            n_samples: Number of random embeddings to test
        """
        logger.note(f"> Test 1: Random embeddings with different values")
        np.random.seed(seed)
        test_embs = np.random.randn(n_samples, EMB_DIM).astype(np.float32)

        logger.mesg(f"  * Test embeddings stats:")
        for i in range(n_samples):
            logger.mesg(
                f"    - emb[{i}] mean={test_embs[i].mean():.4f}, std={test_embs[i].std():.4f}, "
                f"min={test_embs[i].min():.4f}, max={test_embs[i].max():.4f}"
            )

        hash_codes = self.inference.emb_to_hash(test_embs)

        # Calculate pairwise Hamming distances
        hamming_distances = {}
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.sum(hash_codes[i] != hash_codes[j])
                hamming_distances[f"{i}_{j}"] = dist

        info_dict = {
            "input_shape": test_embs.shape,
            "hash_shape": hash_codes.shape,
            "hash_dtype": str(hash_codes.dtype),
            "sample_bits_0": hash_codes[0][:20].tolist(),
            "sample_bits_1": hash_codes[1][:20].tolist(),
            "sample_bits_2": hash_codes[2][:20].tolist(),
        }
        info_dict.update({f"hamming_{k}": v for k, v in hamming_distances.items()})
        logger.mesg(dict_to_str(info_dict), indent=2)

        # Show first 64 bits comparison
        logger.note("> First 64 bits comparison:")
        for i in range(n_samples):
            bits_str = "".join(str(b) for b in hash_codes[i][:64])
            logger.mesg(f"  - [{i}]: {bits_str}")

        # Show hex strings
        hex_strs = self.inference.emb_to_hex(test_embs)
        logger.note(f"> Hex strings: {len(hex_strs)}")
        for i, h in enumerate(hex_strs):
            logger.mesg(f"  - [{i}]: {h[:64]}... (len={len(h)})")

        # Show hex differences
        logger.note("> Hex string differences:")
        for i in range(len(hex_strs)):
            for j in range(i + 1, len(hex_strs)):
                diff_positions = [
                    k
                    for k in range(len(hex_strs[i]))
                    if hex_strs[i][k] != hex_strs[j][k]
                ]
                logger.mesg(
                    f"  * [{i}] vs [{j}]: {len(diff_positions)} chars differ at positions {diff_positions[:10]}..."
                )

    def test_batch_vs_individual(self, seed: int = 42, n_samples: int = 3):
        """Test consistency between batch and individual processing.

        Args:
            seed: Random seed for reproducibility
            n_samples: Number of embeddings to test
        """
        logger.note("> Test 2: Process embeddings one by one")
        np.random.seed(seed)
        test_embs = np.random.randn(n_samples, EMB_DIM).astype(np.float32)

        # Batch processing
        hash_codes_batch = self.inference.emb_to_hash(test_embs)

        # Individual processing
        hash_codes_individual = []
        for i in range(n_samples):
            h = self.inference.emb_to_hash(test_embs[i : i + 1])
            hash_codes_individual.append(h[0])
        hash_codes_individual = np.array(hash_codes_individual)

        # Compare
        batch_vs_individual = np.sum(hash_codes_batch != hash_codes_individual)
        logger.mesg(f"  * Batch vs Individual differences: {batch_vs_individual} bits")
        if batch_vs_individual == 0:
            logger.okay("  ✓ Batch and individual processing are consistent")
        else:
            logger.warn(f"  ✗ Found {batch_vs_individual} bit differences!")

    def test_extreme_separation(self):
        """Test separation between very different embeddings.

        Tests with:
        - All zeros vector
        - All ones vector
        - Random normal vector

        Healthy models should show strong separation (>30% Hamming distance).
        """
        logger.note("> Test 3: Very different embeddings (zeros, ones, random)")
        diverse_embs = np.array(
            [
                np.zeros(EMB_DIM, dtype=np.float32),
                np.ones(EMB_DIM, dtype=np.float32),
                np.random.randn(EMB_DIM).astype(np.float32),
            ]
        )
        diverse_hash = self.inference.emb_to_hash(diverse_embs)

        hamming_01 = np.sum(diverse_hash[0] != diverse_hash[1])
        hamming_02 = np.sum(diverse_hash[0] != diverse_hash[2])
        hamming_12 = np.sum(diverse_hash[1] != diverse_hash[2])

        total_bits = diverse_hash.shape[1]
        logger.mesg(
            f"  * Hamming distances: "
            f"0-1={hamming_01} ({hamming_01/total_bits*100:.1f}%), "
            f"0-2={hamming_02} ({hamming_02/total_bits*100:.1f}%), "
            f"1-2={hamming_12} ({hamming_12/total_bits*100:.1f}%)"
        )

        # Check if model is healthy (>30% separation)
        min_distance = min(hamming_01, hamming_02, hamming_12)
        if min_distance > total_bits * 0.3:
            logger.okay(
                f"  ✓ Model shows good separation (min {min_distance/total_bits*100:.1f}%)"
            )
        else:
            logger.warn(
                f"  ✗ Model shows poor separation (min {min_distance/total_bits*100:.1f}%), "
                "expected >30%. Model may need retraining."
            )

    def run_all_tests(self):
        """Run all test suites."""
        logger.note("> Testing HasherInference")
        self.test_random_embeddings()
        self.test_batch_vs_individual()
        self.test_extreme_separation()


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
        self.add_argument(
            "-rm",
            "--remove-weights",
            action="store_true",
            help="Remove existing weights before training",
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
            remove_weights=args.remove_weights,
        )
        trainer.init_model()
        trainer.init_dataloader()
        trainer.train(epochs=args.epochs, resume=not args.overwrite)

    elif args.mode == "test":
        tester = HasherInferenceTester()
        tester.run_all_tests()


if __name__ == "__main__":
    main()
    # Case: mode train
    # python -m models.tembed.hasher -m train -hb 2048 -hd 1024 -ms 50000 -ep 10
    # python -m models.tembed.hasher -m train -hb 2048 -ms 1000000 -ep 50
    # python -m models.tembed.hasher -m train -ep 50 -dp 0.2
    # python -m models.tembed.hasher -m train -w

    # Case: mode test
    # python -m models.tembed.hasher -m test
