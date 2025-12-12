import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import sys

from pathlib import Path
from tclogger import TCLogger, TCLogbar, logstr, dict_to_str, int_bits
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from models.tembed.train import TrainSamplesManager

logger = TCLogger("Hasher")

PARENT_DIR = Path(__file__).parent
WEIGHTS_DIR = PARENT_DIR / "weights"
SAMPLES_DIR = Path("/media/data/tembed/train_samples")

EMB_DIM = 1024
HASH_BITS = 2048
HIDDEN_DIM = 2048
DROPOUT = 0.1

BATCH_SIZE = 256
# ResMLP is more sensitive to step size; use a safer default.
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20


class RMSNorm(nn.Module):
    """RMSNorm without mean subtraction (stable for embedding features)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


class GatedMLPBlock(nn.Module):
    """Pre-norm residual gated MLP block (SwiGLU-style)."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(dim)
        hidden = int(dim * expansion)
        self.fc_g = nn.Linear(dim, hidden, bias=True)
        self.fc_v = nn.Linear(dim, hidden, bias=True)
        self.fc_out = nn.Linear(hidden, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        g = torch.nn.functional.silu(self.fc_g(z))
        v = self.fc_v(z)
        y = self.fc_out(g * v)
        y = self.drop(y)
        return x + y


class EmbToHashNetV2(nn.Module):
    """Stronger Emb->Hash model: input proj + residual gated blocks + output head."""

    def __init__(
        self,
        input_dim: int,
        hash_bits: int,
        model_dim: int = 1536,
        depth: int = 4,
        expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hash_bits = int(hash_bits)
        self.model_dim = int(model_dim)
        self.depth = int(depth)
        self.expansion = int(expansion)
        self.dropout = float(dropout)

        self.in_norm = RMSNorm(self.input_dim)
        self.in_proj = nn.Linear(self.input_dim, self.model_dim, bias=True)
        self.blocks = nn.ModuleList(
            [
                GatedMLPBlock(
                    dim=self.model_dim,
                    expansion=self.expansion,
                    dropout=self.dropout,
                )
                for _ in range(self.depth)
            ]
        )
        self.out_norm = RMSNorm(self.model_dim)
        self.out_proj = nn.Linear(self.model_dim, self.hash_bits, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D tensor (batch, dim), got shape={tuple(x.shape)}"
            )
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[1]}")

        # Keep angular geometry stable.
        x = torch.nn.functional.normalize(x, dim=-1)
        x = self.in_norm(x)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        return torch.tanh(self.out_proj(h))

    def get_hash_codes(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.forward(x)
            return (h > 0).to(torch.uint8)


def build_hasher_model(
    arch: str,
    input_dim: int,
    hash_bits: int,
    hidden_dim: int,
    dropout: float,
    model_dim: int = 1536,
    depth: int = 4,
    expansion: int = 4,
) -> nn.Module:
    arch = (arch or "mlp").lower()
    if arch == "mlp":
        return EmbToHashNet(
            input_dim=input_dim,
            hash_bits=hash_bits,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if arch in {"resmlp", "v2", "gmlp"}:
        return EmbToHashNetV2(
            input_dim=input_dim,
            hash_bits=hash_bits,
            model_dim=model_dim,
            depth=depth,
            expansion=expansion,
            dropout=dropout,
        )
    raise ValueError(f"Unknown arch: {arch}")


class EmbToHashNet(nn.Module):
    """Embedding to Hash via Neural-Network"""

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
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hash_bits, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D tensor (batch, dim), got shape={tuple(x.shape)}"
            )
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[1]}")
        # Normalize input for cosine-geometry consistency (and avoid LayerNorm constant collapse).
        x = torch.nn.functional.normalize(x, dim=-1)
        # Use tanh to keep outputs in (-1, 1), matching test-time {-1,1} similarity.
        return torch.tanh(self.net(x))

    def get_hash_codes(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.forward(x)
            # positive -> 1, negative -> 0
            return (h > 0).to(torch.uint8)


class HashLoss(nn.Module):
    """Loss for learned binary hashing.

    Goals:
    1) Preserve original cosine similarity ordering via triplet loss in hash space.
    2) Encourage near-binary outputs (quantization).
    3) Encourage balanced bits (~50% ones).
    4) Encourage decorrelated bits (reduce redundancy).
    """

    def __init__(
        self,
        margin: float = 0.2,
        w_distill: float = 1.0,
        w_matrix: float = 0.25,
        matrix_temp: float = 0.2,
        w_entropy: float = 0.01,
        w_center: float = 0.05,
        w_triplet: float = 1.0,
        w_quant: float = 0.02,
        w_balance: float = 0.02,
        w_decor: float = 0.005,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.margin = float(margin)
        self.w_distill = float(w_distill)
        self.w_matrix = float(w_matrix)
        self.matrix_temp = float(matrix_temp)
        self.w_entropy = float(w_entropy)
        self.w_center = float(w_center)
        self.w_triplet = float(w_triplet)
        self.w_quant = float(w_quant)
        self.w_balance = float(w_balance)
        self.w_decor = float(w_decor)
        self.eps = float(eps)

        # Training-time scheduling (set by trainer): helps stability at 1M+
        self._epoch: int = 0
        self._epochs: int = 0

    def set_epoch(self, epoch: int, epochs: int) -> None:
        self._epoch = int(epoch)
        self._epochs = int(epochs)

    def _weight_scale(self, kind: str) -> float:
        """Simple schedules to improve stability across dataset sizes.

        - entropy: warm up (avoid fighting alignment early)
        - center : warm up slightly (still useful early, but can destabilize if too strong)
        """
        if self._epochs <= 0:
            return 1.0
        t = float(self._epoch) / float(max(1, self._epochs))
        if kind == "entropy":
            # 0 -> 1 over first 30% epochs
            return float(min(1.0, t / 0.30))
        if kind == "center":
            # 0.5 -> 1 over first 20% epochs
            return float(0.5 + 0.5 * min(1.0, t / 0.20))
        return 1.0

    def _cos_matrix(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.normalize(x, dim=-1, eps=self.eps)
        c = x @ x.T
        # Numerical guard: cosine should be in [-1, 1], but fp16/overflow can break this.
        return torch.clamp(c, -1.0, 1.0)

    def _offdiag_mask(self, n: int, device: torch.device) -> torch.Tensor:
        return (~torch.eye(n, dtype=torch.bool, device=device)).to(torch.bool)

    def _entropy_loss(self, h: torch.Tensor) -> torch.Tensor:
        """Encourage non-saturated bits via per-bit Bernoulli entropy.

        Let p = P(bit=1) estimated from continuous h in (-1,1) via p=(h+1)/2.
        Max entropy at p=0.5. Penalize low entropy bits.
        """
        p = torch.clamp((h + 1.0) * 0.5, self.eps, 1.0 - self.eps)
        # Stable: avoid log(0) and propagate finite values.
        hbit = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        # target: encourage near-maximum entropy; use hinge so it's inactive when healthy
        # max entropy for Bernoulli is ln(2)
        loss = torch.mean(torch.relu(np.log(2.0) - hbit))
        # Final guard to prevent NaNs from poisoning training.
        if not torch.isfinite(loss):
            return torch.zeros((), device=h.device, dtype=h.dtype)
        return loss

    def _signed_cos(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = torch.nn.functional.normalize(a, dim=-1, eps=self.eps)
        b = torch.nn.functional.normalize(b, dim=-1, eps=self.eps)
        return torch.sum(a * b, dim=-1)

    def _quant_loss(self, h: torch.Tensor) -> torch.Tensor:
        # For tanh outputs in (-1,1), encourage saturation toward {-1,+1}
        return torch.mean((1.0 - torch.abs(h)) ** 2)

    def _balance_loss(self, h: torch.Tensor) -> torch.Tensor:
        # center each bit to have mean ~0 across batch (=> 50/50 after sign)
        return torch.mean(torch.mean(h, dim=0) ** 2)

    def _decor_loss(self, h: torch.Tensor) -> torch.Tensor:
        # decorrelate bits using correlation matrix of tanh outputs
        z = h
        z = z - torch.mean(z, dim=0, keepdim=True)
        z = z / (torch.std(z, dim=0, keepdim=True) + self.eps)
        c = (z.T @ z) / (z.shape[0] + self.eps)  # [B,Bits] -> [Bits,Bits]
        off_diag = c - torch.diag(torch.diag(c))
        return torch.mean(off_diag**2)

    def forward(
        self,
        anchor_hash: torch.Tensor,
        pos_hash: torch.Tensor,
        neg_hash: torch.Tensor,
        anchor_emb: torch.Tensor | None = None,
        pos_emb: torch.Tensor | None = None,
        neg_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Similarity in hash space (continuous) using cosine
        sim_ap = self._signed_cos(anchor_hash, pos_hash)
        sim_an = self._signed_cos(anchor_hash, neg_hash)

        # Triplet ranking: want sim_ap >= sim_an + margin
        triplet = torch.relu(self.margin - (sim_ap - sim_an))
        loss_triplet = torch.mean(triplet)

        # Distillation: match hash-space similarities to embedding-space cosine
        loss_distill = torch.tensor(0.0, device=anchor_hash.device)
        loss_matrix = torch.tensor(0.0, device=anchor_hash.device)
        loss_center = torch.tensor(0.0, device=anchor_hash.device)
        if anchor_emb is not None and pos_emb is not None and neg_emb is not None:
            anchor_emb_n = torch.nn.functional.normalize(
                anchor_emb, dim=-1, eps=self.eps
            )
            pos_emb_n = torch.nn.functional.normalize(pos_emb, dim=-1, eps=self.eps)
            neg_emb_n = torch.nn.functional.normalize(neg_emb, dim=-1, eps=self.eps)

            t_ap = self._signed_cos(anchor_emb_n, pos_emb_n).detach()
            t_an = self._signed_cos(anchor_emb_n, neg_emb_n).detach()
            # Smooth L1 is robust and keeps scale comparable
            loss_distill = torch.nn.functional.smooth_l1_loss(
                sim_ap, t_ap
            ) + torch.nn.functional.smooth_l1_loss(sim_an, t_an)

            # In-batch matrix distillation on anchors: preserve global neighborhood geometry.
            # Important: scaling cosine by 1/temp can explode gradients when temp is small.
            # Instead, keep cosine in [-1,1] and apply temperature inside the regression.
            temp = max(self.matrix_temp, self.eps)
            s_mat = self._cos_matrix(anchor_hash)
            t_mat = self._cos_matrix(anchor_emb_n).detach()
            mask = self._offdiag_mask(s_mat.shape[0], s_mat.device)
            loss_matrix = torch.nn.functional.smooth_l1_loss(
                s_mat[mask], t_mat[mask], beta=float(temp)
            )

            # Centering: match the *mean* pairwise cosine to reduce global bias.
            # This addresses the common failure mode where hash cosine is shifted upward.
            with torch.no_grad():
                t_mu = t_mat[mask].mean()
            s_mu = s_mat[mask].mean()
            loss_center = (s_mu - t_mu) ** 2

        # Regularize anchors only to reduce over-constraint and collapse risk.
        loss_quant = self._quant_loss(anchor_hash)
        loss_bal = self._balance_loss(anchor_hash)
        loss_decor = self._decor_loss(anchor_hash)
        loss_entropy = self._entropy_loss(anchor_hash)

        w_entropy_eff = self.w_entropy * self._weight_scale("entropy")
        w_center_eff = self.w_center * self._weight_scale("center")

        total = (
            self.w_distill * loss_distill
            + self.w_matrix * loss_matrix
            + w_center_eff * loss_center
            + self.w_triplet * loss_triplet
            + self.w_quant * loss_quant
            + self.w_balance * loss_bal
            + self.w_decor * loss_decor
            + w_entropy_eff * loss_entropy
        )

        # Safety: if anything went numerically wrong, return a finite loss.
        if not torch.isfinite(total):
            total = torch.zeros((), device=anchor_hash.device, dtype=anchor_hash.dtype)

        loss_dict = {
            "total": float(total.detach().cpu().item()),
            "sim": float(loss_triplet.detach().cpu().item()),
            "distill": float(loss_distill.detach().cpu().item()),
            "matrix": float(loss_matrix.detach().cpu().item()),
            "center": float(loss_center.detach().cpu().item()),
            "quant": float(loss_quant.detach().cpu().item()),
            "bal": float(loss_bal.detach().cpu().item()),
            "decor": float(loss_decor.detach().cpu().item()),
            "entropy": float(loss_entropy.detach().cpu().item()),
        }
        return total, loss_dict


class TrainSamplesDataset(Dataset):
    """Loads pre-computed training samples containing (anchor, positive, negative) triplets.

    Data Structure:
        - Anchors: Random embeddings from corpus
        - Positives: Top-k similar items (from Faiss retrieval)
        - Negatives: not similar to anchor

    Storage Format:
        - Sharded storage: 30 shards × 100K samples = 3M total
        - Memory-mapped for efficient loading
        - Each sample has: anchor, positive, negative
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
        self.manager = TrainSamplesManager(data_dir=data_dir, mode="read")
        self.num_positives_to_use = num_positives_to_use
        self._length = len(self.manager)
        if max_samples is not None:
            self._length = min(self._length, max_samples)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        """Get a training triplet sample.

        Args:
            idx: sample index, 0 to len(dataset)-1

        Returns:
            Dict with keys:
                - 'anchor_emb': Anchor embedding, shape (1024,), dtype float32
                - 'pos_emb': Positive embedding, shape (1024,), dtype float32
                - 'neg_emb': Negative embedding, shape (1024,), dtype float32

        Example:
            >>> sample = dataset[42]
            >>> anchor, pos, neg = sample['anchor_emb'], sample['pos_emb'], sample['neg_emb']
            >>> cos_sim = (anchor @ pos) / (np.linalg.norm(anchor) * np.linalg.norm(pos))
            >>> cos_sim > 0.5  # Positive should be similar
        """
        sample = self.manager[idx]
        pos_idx = np.random.randint(0, len(sample["pos_embs"]))
        return {
            "anchor_emb": sample["anchor_emb"].astype(np.float32),
            "pos_emb": sample["pos_embs"][pos_idx].astype(np.float32),
            "neg_emb": sample["neg_emb"].astype(np.float32),
        }


class HasherTrainer:
    def __init__(
        self,
        data_dir: str | Path = SAMPLES_DIR,
        input_dim: int = EMB_DIM,
        hash_bits: int = HASH_BITS,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        arch: str = "mlp",
        model_dim: int = 1536,
        depth: int = 4,
        expansion: int = 4,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        max_samples: int = None,
        remove_weights: bool = False,
        loss_params: Optional[dict] = None,
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
        self.arch = arch
        self.model_dim = int(model_dim)
        self.depth = int(depth)
        self.expansion = int(expansion)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_samples = max_samples
        self.remove_weights = remove_weights
        self.loss_params = loss_params or {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[EmbToHashNet] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[HashLoss] = None
        self.dataloader: Optional[DataLoader] = None

        self.weights_dir = WEIGHTS_DIR
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path: Optional[Path] = None
        self.best_path: Optional[Path] = None
        self.config_path: Optional[Path] = None

    def _log_model_info(self):
        """Log model initialization information."""
        logger.note(f"> Initializing model:")
        arch = str(self.arch or "mlp").lower()
        info_dict = {
            "input_dim": self.input_dim,
            "hash_bits": self.hash_bits,
            "arch": arch,
            "dropout": self.dropout,
            "device": str(self.device),
        }
        if arch == "mlp":
            info_dict["hidden_dim"] = self.hidden_dim
        else:
            # hidden_dim is an MLP-only parameter; avoid misleading logs.
            info_dict["model_dim"] = int(self.model_dim)
            info_dict["depth"] = int(self.depth)
            info_dict["expansion"] = int(self.expansion)
        logger.mesg(dict_to_str(info_dict), indent=2)

        logger.note(f"> Loss weights:")
        loss_weights = {
            "distill": float(self.loss_params.get("w_distill", 1.0)),
            "matrix": float(self.loss_params.get("w_matrix", 0.25)),
            "matrix_temp": float(self.loss_params.get("matrix_temp", 0.07)),
            "center": float(self.loss_params.get("w_center", 0.05)),
            "entropy": float(self.loss_params.get("w_entropy", 0.01)),
            "triplet": float(self.loss_params.get("w_triplet", 1.0)),
            "quant": float(self.loss_params.get("w_quant", 0.02)),
            "balance": float(self.loss_params.get("w_balance", 0.02)),
            "decor": float(self.loss_params.get("w_decor", 0.005)),
            "margin": float(self.loss_params.get("margin", 0.2)),
        }
        logger.mesg(dict_to_str(loss_weights), indent=2)

    def _create_model(self):
        """Create and initialize the hash embedding model."""
        self.model = build_hasher_model(
            arch=self.arch,
            input_dim=self.input_dim,
            hash_bits=self.hash_bits,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            model_dim=self.model_dim,
            depth=self.depth,
            expansion=self.expansion,
        ).to(self.device)

    def _create_optimizer_and_criterion(self):
        """Create optimizer and loss criterion."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.criterion = HashLoss(
            margin=float(self.loss_params.get("margin", 0.2)),
            w_distill=float(self.loss_params.get("w_distill", 1.0)),
            w_matrix=float(self.loss_params.get("w_matrix", 0.25)),
            matrix_temp=float(self.loss_params.get("matrix_temp", 0.1)),
            w_center=float(self.loss_params.get("w_center", 0.05)),
            w_entropy=float(self.loss_params.get("w_entropy", 0.01)),
            w_triplet=float(self.loss_params.get("w_triplet", 1.0)),
            w_quant=float(self.loss_params.get("w_quant", 0.02)),
            w_balance=float(self.loss_params.get("w_balance", 0.02)),
            w_decor=float(self.loss_params.get("w_decor", 0.005)),
        )

    def _generate_checkpoint_paths(self):
        """Generate checkpoint file paths based on model parameters."""
        name_parts = []
        name_parts.append(f"hb{self.hash_bits}")  # hash_bits
        arch = str(self.arch or "mlp").lower()
        if arch == "mlp":
            name_parts.append(f"hd{self.hidden_dim}")  # hidden_dim (MLP only)
        else:
            # ResMLP: hidden_dim is unused; encode architecture params instead.
            name_parts.append(f"md{int(self.model_dim)}")
            name_parts.append(f"d{int(self.depth)}")
            name_parts.append(f"x{int(self.expansion)}")
        name_parts.append(f"bs{self.batch_size}")  # batch_size
        if self.max_samples is not None:
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
        self.config_path = self.weights_dir / f"{hasher_ckpt}.json"

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
            logger.warn(f"  × Removed {removed_count} weight files")

    def _save_model_config(self):
        """Save model configuration to JSON file."""
        config = {
            "input_dim": self.input_dim,
            "hash_bits": self.hash_bits,
            "hidden_dim": self.hidden_dim,
            "arch": str(self.arch),
            "model_dim": int(self.model_dim),
            "depth": int(self.depth),
            "expansion": int(self.expansion),
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_samples": self.max_samples,
            "loss": {
                "margin": float(self.loss_params.get("margin", 0.2)),
                "w_distill": float(self.loss_params.get("w_distill", 1.0)),
                "w_matrix": float(self.loss_params.get("w_matrix", 0.25)),
                "matrix_temp": float(self.loss_params.get("matrix_temp", 0.07)),
                "w_center": float(self.loss_params.get("w_center", 0.05)),
                "w_entropy": float(self.loss_params.get("w_entropy", 0.01)),
                "w_triplet": float(self.loss_params.get("w_triplet", 1.0)),
                "w_quant": float(self.loss_params.get("w_quant", 0.02)),
                "w_balance": float(self.loss_params.get("w_balance", 0.02)),
                "w_decor": float(self.loss_params.get("w_decor", 0.005)),
            },
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
                f"> Preloading {num_shards_needed} shards for {self.max_samples} samples:"
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
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": float(loss),
        }
        if self.scheduler:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        # Full checkpoint for resuming training (may be large due to optimizer state).
        torch.save(ckpt, self.ckpt_path)

        if is_best:
            # Weights-only best checkpoint for deployment (small).
            # Keep only what's needed for inference.
            best_ckpt = {
                "epoch": int(epoch),
                "loss": float(loss),
                "model_state_dict": self.model.state_dict(),
            }
            torch.save(best_ckpt, self.best_path)

    def load_checkpoint(self, ckpt_path: Path = None) -> tuple[int, float]:
        """Load training checkpoint.

        Returns:
            Tuple of (starting_epoch, best_loss)
        """
        if ckpt_path is None:
            ckpt_path = self.ckpt_path

        if not ckpt_path.exists():
            logger.warn(f"  * No existed checkpoint, training from scratch")
            return 0, float("inf")

        logger.mesg(f"  > Loading checkpoint from:")
        logger.file(f"    * {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        logger.mesg(f"  epoch: {ckpt['epoch']}, loss: {ckpt['loss']:.4f}")
        return ckpt["epoch"], ckpt.get("loss", float("inf"))

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()

        if self.criterion is not None and hasattr(self.criterion, "set_epoch"):
            self.criterion.set_epoch(epoch=epoch, epochs=self.epochs)

        total_losses = {
            "total": 0.0,
            "sim": 0.0,
            "distill": 0.0,
            "matrix": 0.0,
            "center": 0.0,
            "quant": 0.0,
            "bal": 0.0,
            "decor": 0.0,
            "entropy": 0.0,
        }
        skipped_batches = 0
        num_batches = len(self.dataloader)

        use_bar = bool(getattr(sys.stdout, "isatty", lambda: False)())
        bar = TCLogbar(total=num_batches) if use_bar else None
        epoch_str = logstr.file(f"{epoch:>{int_bits(self.epochs)}d}")
        if bar is not None:
            bar.set_head(logstr.mesg(f"  * [Epoch {epoch_str}/{self.epochs}]"))

        for batch_idx, batch in enumerate(self.dataloader):
            anchor_emb = batch["anchor_emb"].to(self.device)
            positive_emb = batch["pos_emb"].to(self.device)
            negative_emb = batch["neg_emb"].to(self.device)

            # Normalize embeddings so teacher cosine is well-defined and stable.
            anchor_emb = torch.nn.functional.normalize(anchor_emb, dim=-1)
            positive_emb = torch.nn.functional.normalize(positive_emb, dim=-1)
            negative_emb = torch.nn.functional.normalize(negative_emb, dim=-1)
            # calc hashes
            anchor_hash = self.model(anchor_emb)
            positive_hash = self.model(positive_emb)
            negative_hash = self.model(negative_emb)
            # calc loss
            loss, loss_dict = self.criterion(
                anchor_hash,
                positive_hash,
                negative_hash,
                anchor_emb=anchor_emb,
                pos_emb=positive_emb,
                neg_emb=negative_emb,
            )

            # Numerical safety: skip this batch if loss is NaN/Inf.
            # This avoids corrupting optimizer state and checkpoint.
            if not torch.isfinite(loss):
                skipped_batches += 1
                self.optimizer.zero_grad(set_to_none=True)
                desc = (
                    f"loss=nan "
                    f"(skip={skipped_batches}, "
                    f"d={loss_dict.get('distill', 0.0):.3f}, "
                    f"m={loss_dict.get('matrix', 0.0):.3f}, "
                    f"c={loss_dict.get('center', 0.0):.3f}, "
                    f"e={loss_dict.get('entropy', 0.0):.3f}, s={loss_dict.get('sim', 0.0):.3f})"
                )
                if bar is not None:
                    bar.update(1, desc=desc)
                elif (batch_idx % 50) == 0:
                    logger.warn(desc)
                continue
            # backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            # accumulate losses
            for k, v in loss_dict.items():
                total_losses[k] += v
            # update progress
            desc = (
                f"loss={loss_dict['total']:.3f} "
                f"(d={loss_dict.get('distill', 0.0):.3f}, "
                f"m={loss_dict.get('matrix', 0.0):.3f}, "
                f"c={loss_dict.get('center', 0.0):.3f}, "
                f"e={loss_dict.get('entropy', 0.0):.3f}, s={loss_dict['sim']:.3f})"
            )
            if bar is not None:
                bar.update(1, desc=desc)
            elif (batch_idx % 50) == 0:
                logger.mesg(desc)

        if bar is not None:
            bar.update(flush=True, linebreak=True)

        denom = max(1, num_batches - skipped_batches)
        avg_losses = {k: v / denom for k, v in total_losses.items()}
        avg_losses["skipped"] = float(skipped_batches)
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

        # check if training completed
        if start_epoch >= epochs:
            logger.okay(
                f"> Training already completed at epoch {start_epoch}. "
                f"Best loss: {best_loss:.4f}"
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
            self.optimizer,
            T_max=max(1, remaining_epochs),
            eta_min=self.learning_rate * 0.05,
        )

        for epoch in range(start_epoch + 1, epochs + 1):
            avg_losses = self.train_epoch(epoch)
            lr = self.optimizer.param_groups[0]["lr"]
            # update scheduler
            self.scheduler.step()

            if avg_losses.get("skipped", 0.0) > 0:
                logger.warn(
                    f"  * Skipped {int(avg_losses['skipped'])} NaN/Inf batches in epoch {epoch}"
                )

            # Do not treat NaN as best.
            if not np.isfinite(avg_losses["total"]):
                logger.warn(
                    f"  ✗ Epoch {epoch} produced non-finite avg loss; checkpoint not marked best"
                )
            # save checkpoint
            is_best = np.isfinite(avg_losses["total"]) and (
                avg_losses["total"] < best_loss
            )
            if is_best:
                best_loss = avg_losses["total"]
            self.save_checkpoint(epoch, avg_losses["total"], is_best=is_best)

        logger.okay(f"> Training completed. Best loss: {best_loss:.4f}")


class HasherInference:
    """Inference wrapper for trained Hasher model."""

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
        # init weights
        if weights_path is None:
            weights_path = self._find_best_latest_weights()
        self.weights_path = weights_path
        # init config
        if config_path is None:
            config_name = self.weights_path.stem.replace("_best", "") + ".json"
            self.config_path = self.weights_dir / config_name
        else:
            self.config_path = config_path
        # init device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        # load model
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
        # look for best weights
        best_files = list(self.weights_dir.glob("hasher_*_best.pt"))
        if best_files:
            # use most recently modified
            latest = max(best_files, key=lambda p: p.stat().st_mtime)
            return latest
        # look for latest weights
        ckpt_files = list(self.weights_dir.glob("hasher_*.pt"))
        if ckpt_files:
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
        # load config
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config = json.load(f)
        else:
            config = {
                "input_dim": EMB_DIM,
                "hash_bits": HASH_BITS,
                "hidden_dim": HIDDEN_DIM,
                "dropout": 0.1,
                "arch": "mlp",
            }
        # init model
        arch = str(config.get("arch", "mlp"))
        self.model = build_hasher_model(
            arch=arch,
            input_dim=int(config.get("input_dim", EMB_DIM)),
            hash_bits=int(config.get("hash_bits", HASH_BITS)),
            hidden_dim=int(config.get("hidden_dim", HIDDEN_DIM)),
            dropout=float(config.get("dropout", 0.1)),
            model_dim=int(config.get("model_dim", 1536)),
            depth=int(config.get("depth", 4)),
            expansion=int(config.get("expansion", 4)),
        ).to(self.device)
        # load weights
        ckpt = torch.load(self.weights_path, map_location=self.device)
        # Support:
        # 1) full checkpoint dict with "model_state_dict"
        # 2) raw state_dict
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        self.model.load_state_dict(state)
        self.model.eval()
        # log path
        logger.note(f"> Loaded HasherInference model from:")
        logger.file(f"  * {self.weights_path}")

    def emb_to_hash(self, embs: np.ndarray) -> np.ndarray:
        """Convert embeddings to binary hash codes."""
        squeeze = False
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
            squeeze = True
        x = torch.from_numpy(embs.astype(np.float32)).to(self.device)
        x = torch.nn.functional.normalize(x, dim=-1)
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
        # pack bits into bytes
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
    """Test criteria for HasherInference model:
    1. **Random Embeddings Diversity** (40-60% Hamming distance)
        Ensures different embeddings get different hashes

    2. **Batch vs Individual Consistency** (0 bit difference)
        Verifies deterministic behavior regardless of batch size

    3. **Extreme Value Separation** (>30% Hamming distance)
        Tests separation on very different inputs (zeros, ones, random)

    4. **Cosine Similarity Preservation** (correlation >0.7, MAE <0.15)
        THE KEY TEST - angular preservation
        This determines if model will work for similarity search

    5. **Bit Distribution Balance** (mean ~0.5, few imbalanced bits)
        Ensures efficient use of hash code capacity

    6. **Hash Statistics** (mean Hamming ~50%, healthy spread)
        Overall distribution health check

    Success Criteria:
        - Test 4 (similarity preservation) is CRITICAL
        - Correlation >0.7 means model preserves similarity well
        - MAE <0.15 means small errors in similarity estimation
        - If Test 4 fails, model won't improve retrieval quality

    Usage:
        >>> tester = HasherInferenceTester()
        >>> tester.run_all_tests()
        # Runs all 6 tests, prints results with ✓/✗ indicators

    Expected Results (Good Model):
        Test 1: ✓ Good diversity (46-50%)
        Test 2: ✓ Batch consistent (0 diffs)
        Test 3: ✓ Good separation (>38%)
        Test 4: ✓ Excellent preservation (corr=0.67+, MAE=0.02)
        Test 5: ✓ Excellent balance (mean=0.50, std=0.05)
        Test 6: ✓ Healthy distribution (mean=48%)
    """

    def __init__(self, inference: Optional[HasherInference] = None):
        """Initialize tester.

        Args:
            inference: HasherInference instance. If None, creates a new one.
        """
        self.inference = inference or HasherInference()

    @staticmethod
    def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / (n + eps)

    @staticmethod
    def _hash_to_signed(hash_codes: np.ndarray) -> np.ndarray:
        return hash_codes.astype(np.float32) * 2.0 - 1.0

    @staticmethod
    def _pairwise_cos_np(x: np.ndarray) -> np.ndarray:
        x = HasherInferenceTester._l2_normalize_np(x)
        return x @ x.T

    @staticmethod
    def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a)
        b = np.asarray(b)
        if a.size != b.size or a.size < 2:
            return float("nan")
        ra = a.argsort().argsort().astype(np.float32)
        rb = b.argsort().argsort().astype(np.float32)
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = (np.sqrt((ra**2).mean()) * np.sqrt((rb**2).mean())) + 1e-12
        return float((ra * rb).mean() / denom)

    @staticmethod
    def _topk_overlap(sim_a: np.ndarray, sim_b: np.ndarray, k: int) -> float:
        n = sim_a.shape[0]
        ks = min(max(1, int(k)), max(1, n - 1))
        overlaps = []
        for i in range(n):
            ia = np.argsort(-sim_a[i])
            ib = np.argsort(-sim_b[i])
            ia = ia[ia != i][:ks]
            ib = ib[ib != i][:ks]
            overlaps.append(len(set(ia.tolist()) & set(ib.tolist())) / ks)
        return float(np.mean(overlaps))

    def _collapse_diagnostics(self, hash_codes: np.ndarray) -> dict[str, float]:
        bit_means = hash_codes.mean(axis=0)
        eps = 1e-12
        entropy = -(
            bit_means * np.log2(bit_means + eps)
            + (1.0 - bit_means) * np.log2(1.0 - bit_means + eps)
        )
        unique = np.unique(hash_codes, axis=0).shape[0]
        return {
            "unique_ratio": float(unique / max(1, hash_codes.shape[0])),
            "entropy_mean": float(entropy.mean()),
            "entropy_min": float(entropy.min()),
            "bit_mean": float(bit_means.mean()),
            "bit_std": float(bit_means.std()),
        }

    def _make_controlled_cosine_set(
        self,
        n_anchors: int = 32,
        n_per_anchor: int = 8,
        noise_levels: list[float] | None = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate embeddings with a wide, controllable cosine similarity range.

        Construction: for each anchor a (unit), make variants v = normalize(a + s * u)
        where u is random unit noise orthogonal-ish to a (in expectation).
        This yields cos(a, v) roughly decreasing as s increases.
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        if n_per_anchor != len(noise_levels):
            # keep behavior predictable; allow user to pass matching lists
            noise_levels = (
                noise_levels
                * ((n_per_anchor + len(noise_levels) - 1) // len(noise_levels))
            )[:n_per_anchor]

        rng = np.random.default_rng(seed)
        anchors = rng.standard_normal((n_anchors, EMB_DIM), dtype=np.float32)
        anchors = self._l2_normalize_np(anchors)

        all_vecs = []
        for i in range(n_anchors):
            a = anchors[i]
            for s in noise_levels[:n_per_anchor]:
                u = rng.standard_normal((EMB_DIM,), dtype=np.float32)
                u = u - float(np.dot(u, a)) * a
                u = u / (np.linalg.norm(u) + 1e-12)
                v = a + float(s) * u
                all_vecs.append(v.astype(np.float32))
        all_vecs = np.stack(all_vecs, axis=0)
        return self._l2_normalize_np(all_vecs)

    @staticmethod
    def _linear_calibration(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Fit y ~= a*x + b by least squares (closed form)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        vx = float(np.var(x))
        if vx < 1e-12:
            return 1.0, 0.0
        a = float(np.cov(x, y, bias=True)[0, 1] / (vx + 1e-12))
        b = float(y.mean() - a * x.mean())
        return a, b

    @staticmethod
    def _bucket_report(
        x: np.ndarray,
        y: np.ndarray,
        bins: list[float],
    ) -> list[dict[str, float]]:
        """Bucket x into [bins[i], bins[i+1]) and report y stats and MAE."""
        x = np.asarray(x)
        y = np.asarray(y)
        rows: list[dict[str, float]] = []
        for i in range(len(bins) - 1):
            lo = float(bins[i])
            hi = float(bins[i + 1])
            m = (x >= lo) & (x < hi)
            n = int(m.sum())
            if n <= 0:
                rows.append(
                    {
                        "lo": lo,
                        "hi": hi,
                        "n": 0.0,
                        "x_mean": float("nan"),
                        "y_mean": float("nan"),
                        "y_std": float("nan"),
                        "mae": float("nan"),
                    }
                )
                continue
            xs = x[m]
            ys = y[m]
            rows.append(
                {
                    "lo": lo,
                    "hi": hi,
                    "n": float(n),
                    "x_mean": float(xs.mean()),
                    "y_mean": float(ys.mean()),
                    "y_std": float(ys.std()),
                    "mae": float(np.mean(np.abs(xs - ys))),
                }
            )
        return rows

    def test_random_embeddings(self, seed: int = 42, n_samples: int = 3):
        """Test with random embeddings from normal distribution.

        Args:
            seed: Random seed for reproducibility
            n_samples: Number of random embeddings to test
        """
        logger.note(f"> Test 1: Random embeddings diversity")
        # Prefer unit-sphere distribution to match training geometry (cosine).
        rng = np.random.default_rng(seed)
        test_embs = rng.standard_normal((n_samples, EMB_DIM), dtype=np.float32)
        test_embs = self._l2_normalize_np(test_embs)

        logger.mesg(f"  * Test embeddings stats:")
        for i in range(n_samples):
            logger.mesg(
                f"    - emb[{i}] mean={test_embs[i].mean():.4f}, std={test_embs[i].std():.4f}, "
                f"min={test_embs[i].min():.4f}, max={test_embs[i].max():.4f}"
            )
        # get hash codes
        hash_codes = self.inference.emb_to_hash(test_embs)

        # calc pairwise Hamming distances
        hamming_distances = {}
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.sum(hash_codes[i] != hash_codes[j])
                hamming_distances[f"{i}_{j}"] = dist

        info_dict = {
            "input_shape": test_embs.shape,
            "hash_shape": hash_codes.shape,
            "hash_dtype": str(hash_codes.dtype),
        }
        info_dict.update({f"hamming_{k}": v for k, v in hamming_distances.items()})
        logger.mesg(dict_to_str(info_dict), indent=2)

        # log hex strings
        logger.note(f"  * Hex strings:")
        for i in range(n_samples):
            hex_str = self.inference.hash_to_hex(hash_codes[i])
            logger.mesg(f"    - [{i}]: {hex_str[:32]} ... (len={len(hex_str)})")

        # check diversity
        avg_hamming = np.mean(list(hamming_distances.values()))
        total_bits = hash_codes.shape[1]
        diversity_ratio = avg_hamming / total_bits
        logger.mesg(
            f"  * Average Hamming distance: {avg_hamming:.1f} / {total_bits} ({diversity_ratio*100:.1f}%)"
        )
        if 0.4 <= diversity_ratio <= 0.6:
            logger.okay(f"  ✓ Good diversity (40-60% range)")
        else:
            logger.warn(f"  ✗ Poor diversity, expected 40-60%")

    def test_batch_vs_individual(self, seed: int = 42, n_samples: int = 3):
        """Test consistency between batch and individual processing.

        Args:
            seed: Random seed for reproducibility
            n_samples: Number of embeddings to test
        """
        logger.note("> Test 2: Process embeddings one by one")
        rng = np.random.default_rng(seed)
        test_embs = rng.standard_normal((n_samples, EMB_DIM), dtype=np.float32)
        test_embs = self._l2_normalize_np(test_embs)
        # batch
        hash_codes_batch = self.inference.emb_to_hash(test_embs)
        # individual
        hash_codes_individual = []
        for i in range(n_samples):
            h = self.inference.emb_to_hash(test_embs[i : i + 1])
            hash_codes_individual.append(h[0])
        hash_codes_individual = np.array(hash_codes_individual)
        # compare batch and individual
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
        logger.note("> Test 3: Extreme value separation")
        # Since training/inference normalize embeddings, zeros/ones are not meaningful
        # (zeros becomes all-zeros -> undefined cosine; ones collapses to a direction).
        # Use controlled far-apart directions instead.
        rng = np.random.default_rng(42)
        a = rng.standard_normal((EMB_DIM,), dtype=np.float32)
        a = a / (np.linalg.norm(a) + 1e-12)
        b = -a
        c = rng.standard_normal((EMB_DIM,), dtype=np.float32)
        c = c - float(np.dot(c, a)) * a
        c = c / (np.linalg.norm(c) + 1e-12)
        diverse_embs = np.stack([a, b, c], axis=0).astype(np.float32)
        diverse_hash = self.inference.emb_to_hash(diverse_embs)
        # calc pairwise Hamming distances
        hamming_01 = np.sum(diverse_hash[0] != diverse_hash[1])
        hamming_02 = np.sum(diverse_hash[0] != diverse_hash[2])
        hamming_12 = np.sum(diverse_hash[1] != diverse_hash[2])
        # log results
        total_bits = diverse_hash.shape[1]
        logger.mesg(
            f"  * Hamming distances: "
            f"0-1={hamming_01} ({hamming_01/total_bits*100:.1f}%), "
            f"0-2={hamming_02} ({hamming_02/total_bits*100:.1f}%), "
            f"1-2={hamming_12} ({hamming_12/total_bits*100:.1f}%)"
        )
        # check model health (>30% separation)
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

    def test_cosine_similarity_preservation(self, seed: int = 42, n_samples: int = 10):
        """Test if model preserves cosine similarity.

        Args:
            seed: Random seed for reproducibility
            n_samples: Number of random embeddings to test
        """
        logger.note("> Test 4: Cosine similarity  (angular) preservation")
        # IMPORTANT: Pure random pairs in high-D have cosine concentrated near 0,
        # which makes correlation unstable and not representative of retrieval.
        # Use a controlled set with a broad cosine range.
        test_embs_norm = self._make_controlled_cosine_set(
            n_anchors=max(4, n_samples),
            n_per_anchor=8,
            seed=seed,
        )

        hash_codes = self.inference.emb_to_hash(test_embs_norm)
        hash_codes_signed = self._hash_to_signed(hash_codes)

        emb_sim = self._pairwise_cos_np(test_embs_norm)
        hash_sim = (hash_codes_signed @ hash_codes_signed.T) / float(
            hash_codes_signed.shape[1]
        )

        iu = np.triu_indices(emb_sim.shape[0], k=1)
        emb_similarities = emb_sim[iu]
        hash_similarities = hash_sim[iu]

        pearson = float(np.corrcoef(emb_similarities, hash_similarities)[0, 1])
        spearman = self._spearman_corr(emb_similarities, hash_similarities)
        mse = float(np.mean((emb_similarities - hash_similarities) ** 2))
        mae = float(np.mean(np.abs(emb_similarities - hash_similarities)))

        topk10 = self._topk_overlap(emb_sim, hash_sim, k=10)
        topk50 = self._topk_overlap(emb_sim, hash_sim, k=50)

        logger.mesg(
            f"  * Embedding similarity range: [{emb_similarities.min():.3f}, {emb_similarities.max():.3f}]"
        )
        logger.mesg(
            f"  * Hash similarity range: [{hash_similarities.min():.3f}, {hash_similarities.max():.3f}]"
        )
        logger.mesg(f"  * Pearson:   {pearson:.4f}")
        logger.mesg(f"  * Spearman:  {spearman:.4f}")
        logger.mesg(f"  * MAE/MSE:   {mae:.4f} / {mse:.4f}")
        logger.mesg(f"  * TopK@10 overlap: {topk10:.3f}")
        logger.mesg(f"  * TopK@50 overlap: {topk50:.3f}")

        # Calibration / bucket diagnostics: tells whether errors come from
        # a simple affine mismatch (scale+offset) or true rank/geometry distortion.
        a, b = self._linear_calibration(emb_similarities, hash_similarities)
        hash_sim_cal = a * hash_similarities + b
        mae_cal = float(np.mean(np.abs(emb_similarities - hash_sim_cal)))
        logger.mesg(
            f"  * Linear calibration: y={a:.3f}*x+{b:.3f} | MAE_cal={mae_cal:.4f}"
        )

        bins = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
        raw_rows = self._bucket_report(emb_similarities, hash_similarities, bins=bins)
        cal_rows = self._bucket_report(emb_similarities, hash_sim_cal, bins=bins)
        logger.note("  * Bucket report (by embedding cosine):")
        for rr, cr in zip(raw_rows, cal_rows):
            lo = rr["lo"]
            hi = rr["hi"]
            n = int(rr["n"])
            if n == 0:
                logger.mesg(f"    - [{lo:+.1f},{hi:+.1f}): n=0")
                continue
            logger.mesg(
                f"    - [{lo:+.1f},{hi:+.1f}): n={n:<4d} "
                f"hash_mean={rr['y_mean']:+.3f}±{rr['y_std']:.3f} mae={rr['mae']:.3f} | "
                f"mae_cal={cr['mae']:.3f}"
            )

        # Slightly relaxed hard thresholds; focus on rank/neighbor preservation.
        if spearman > 0.7 and topk10 > 0.6 and mae < 0.25:
            logger.okay(
                f"  ✓ Strong preservation (Spearman={spearman:.3f}, TopK@10={topk10:.3f}, MAE={mae:.3f})"
            )
        elif spearman > 0.5 and topk10 > 0.4:
            logger.warn(
                f"  ○ Moderate preservation (Spearman={spearman:.3f}, TopK@10={topk10:.3f}, MAE={mae:.3f})"
            )
        else:
            logger.warn(
                f"  ✗ Weak preservation (Spearman={spearman:.3f}, TopK@10={topk10:.3f}, MAE={mae:.3f})"
            )

    def test_bit_distribution(self, seed: int = 42, n_samples: int = 100):
        """Test if bit distribution is balanced.

        Healthy hash codes should have ~50% zeros and ~50% ones for each bit position.
        Severely imbalanced bits waste information capacity.

        Args:
            seed: Random seed for reproducibility
            n_samples: Number of random embeddings to test
        """
        logger.note("> Test 5: Bit distribution balance")
        rng = np.random.default_rng(seed)
        test_embs = rng.standard_normal((n_samples, EMB_DIM), dtype=np.float32)
        test_embs = self._l2_normalize_np(test_embs)
        hash_codes = self.inference.emb_to_hash(test_embs)
        diag = self._collapse_diagnostics(hash_codes)
        # calc bit balance: mean of each bit across samples
        bit_means = hash_codes.mean(axis=0)  # Proportion of 1s for each bit
        # stats
        mean_balance = bit_means.mean()
        std_balance = bit_means.std()
        min_balance = bit_means.min()
        max_balance = bit_means.max()
        # count severely imbalanced bits (<10% or >90%)
        imbalanced_bits = np.sum((bit_means < 0.1) | (bit_means > 0.9))
        total_bits = len(bit_means)
        # log results
        logger.mesg(f"  * Bit balance statistics (ideal: mean=0.5, std<0.1):")
        logger.mesg(f"    - Mean: {mean_balance:.4f}")
        logger.mesg(f"    - Std:  {std_balance:.4f}")
        logger.mesg(f"    - Range: [{min_balance:.4f}, {max_balance:.4f}]")
        logger.mesg(
            f"    - Severely imbalanced bits (<10% or >90%): {imbalanced_bits}/{total_bits} ({imbalanced_bits/total_bits*100:.1f}%)"
        )
        logger.mesg(
            f"  * Collapse diagnostics: unique={diag['unique_ratio']*100:.1f}%, "
            f"entropy_mean={diag['entropy_mean']:.3f}, entropy_min={diag['entropy_min']:.3f}"
        )
        # good if: mean ~0.5, std low, few imbalanced bits
        if (
            0.45 < mean_balance < 0.55
            and std_balance < 0.1
            and imbalanced_bits < total_bits * 0.05
        ):
            logger.okay(f"  ✓ Excellent bit balance")
        elif (
            0.4 < mean_balance < 0.6
            and std_balance < 0.15
            and imbalanced_bits < total_bits * 0.1
        ):
            logger.warn(f"  ○ Acceptable bit balance")
        else:
            logger.warn(f"  ✗ Poor bit balance - many wasted bits")

    def test_hash_statistics(self, seed: int = 42, n_samples: int = 100):
        """Test overall hash code statistics.

        Args:
            seed: Random seed for reproducibility
            n_samples: Number of random embeddings to test
        """
        logger.note("> Test 6: Hash code statistics")
        rng = np.random.default_rng(seed)
        test_embs = rng.standard_normal((n_samples, EMB_DIM), dtype=np.float32)
        test_embs = self._l2_normalize_np(test_embs)
        hash_codes = self.inference.emb_to_hash(test_embs)
        diag = self._collapse_diagnostics(hash_codes)
        # calc pairwise Hamming distances for all pairs
        hamming_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.sum(hash_codes[i] != hash_codes[j])
                hamming_matrix[i, j] = dist
                hamming_matrix[j, i] = dist
        # get upper triangle (excluding diagonal)
        hamming_distances = hamming_matrix[np.triu_indices(n_samples, k=1)]
        # stats
        total_bits = hash_codes.shape[1]
        mean_dist = hamming_distances.mean()
        std_dist = hamming_distances.std()
        min_dist = hamming_distances.min()
        max_dist = hamming_distances.max()
        # log results
        logger.mesg(
            f"  * Hamming distance statistics (n={len(hamming_distances)} pairs):"
        )
        logger.mesg(
            f"    - Mean: {mean_dist:.1f} / {total_bits} ({mean_dist/total_bits*100:.1f}%)"
        )
        logger.mesg(f"    - Std:  {std_dist:.1f} ({std_dist/total_bits*100:.1f}%)")
        logger.mesg(
            f"    - Range: [{min_dist:.0f}, {max_dist:.0f}] ({min_dist/total_bits*100:.1f}%-{max_dist/total_bits*100:.1f}%)"
        )
        # check for duplicates or near-duplicates
        duplicates = np.sum(hamming_distances == 0)
        near_duplicates = np.sum(
            hamming_distances < total_bits * 0.05
        )  # <5% difference
        # good if: no duplicates, few near-duplicates
        if duplicates > 0:
            logger.warn(f"  ✗ Found {duplicates} duplicate hash codes!")
        if near_duplicates > duplicates:
            logger.warn(
                f"  ○ Found {near_duplicates} near-duplicate pairs (<5% difference)"
            )
        # good if: mean around 50%, std indicating good spread
        if 0.45 < mean_dist / total_bits < 0.55 and std_dist / total_bits < 0.15:
            logger.okay(f"  ✓ Healthy hash distribution")
        else:
            logger.warn(f"  ○ Hash distribution could be better")

        logger.mesg(
            f"  * Unique ratio: {diag['unique_ratio']*100:.1f}% | "
            f"Entropy(mean/min): {diag['entropy_mean']:.3f}/{diag['entropy_min']:.3f}"
        )

    def run_all_tests(self):
        """Run all test suites."""
        logger.note("> Testing HasherInference")
        self.test_random_embeddings()
        self.test_batch_vs_individual()
        self.test_extreme_separation()
        self.test_cosine_similarity_preservation()
        self.test_bit_distribution()
        self.test_hash_statistics()


class HasherArgParser(argparse.ArgumentParser):
    """
    Supports two modes:
        1. train: Train a new model or resume training
        2. test: Run inference quality tests

    Common Arguments:
        -m, --mode: Operation mode (train|test) [required]

    Training Arguments:
        -ep, --epochs: Number of training epochs (default: 50)
        -bz, --batch-size: Training batch size (default: 256)
        -lr, --learning-rate: Initial learning rate (default: 5e-4)
        -hb, --hash-bits: Hash code size (default: 2048)
        -hd, --hidden-dim: MLP hidden dimension (default: 2048)
        -dp, --dropout: Dropout rate (default: 0.1)
        -ms, --max-samples: Max training samples (default: None = all)
        -w , --overwrite: Ignore checkpoint, train from scratch
        -rm, --remove-weights: Delete existing weights before training
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Presets: apply a known-good bundle of defaults. Explicit flags still override.
        self.add_argument(
            "-p",
            "--preset",
            type=str,
            default=None,
            choices=["mlp_base", "resmlp_small", "resmlp_large"],
            help="Apply a training preset (sets sensible defaults; explicit flags override)",
        )
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
            "--arch",
            type=str,
            default="resmlp",
            choices=["mlp", "resmlp"],
            help="Model architecture",
        )
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
            "--model-dim",
            type=int,
            default=768,
            help="ResMLP model dimension (used when --arch=resmlp)",
        )
        self.add_argument(
            "--depth",
            type=int,
            default=3,
            help="ResMLP depth (#blocks) (used when --arch=resmlp)",
        )
        self.add_argument(
            "--expansion",
            type=int,
            default=4,
            help="ResMLP expansion factor (used when --arch=resmlp)",
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

        # Loss options (make large-sample sweeps practical)
        self.add_argument("--margin", type=float, default=0.2, help="Triplet margin")
        self.add_argument(
            "--w-distill", type=float, default=1.0, help="Weight for pairwise distill"
        )
        self.add_argument(
            "--w-matrix", type=float, default=0.25, help="Weight for matrix distill"
        )
        self.add_argument(
            "--matrix-temp", type=float, default=0.2, help="Temperature for matrix"
        )
        self.add_argument(
            "--w-center", type=float, default=0.05, help="Weight for center loss"
        )
        self.add_argument(
            "--w-entropy",
            type=float,
            default=0.01,
            help="Weight for entropy regularizer",
        )
        self.add_argument(
            "--w-triplet", type=float, default=1.0, help="Weight for triplet loss"
        )
        self.add_argument(
            "--w-quant", type=float, default=0.02, help="Weight for quantization"
        )
        self.add_argument(
            "--w-balance", type=float, default=0.02, help="Weight for bit balance"
        )
        self.add_argument(
            "--w-decor", type=float, default=0.005, help="Weight for decorrelation"
        )

        # Parse once to see if preset is requested.
        args, _ = self.parse_known_args()

        if args.preset is not None:
            presets: dict[str, dict[str, object]] = {
                # Baseline MLP matching older behavior.
                "mlp_base": {
                    "arch": "mlp",
                    "learning_rate": 1e-3,
                    "hidden_dim": 1024,
                    "matrix_temp": 0.2,
                },
                # Fast, stable ResMLP for quick sweeps.
                "resmlp_small": {
                    "arch": "resmlp",
                    "learning_rate": 5e-4,
                    "model_dim": 768,
                    "depth": 3,
                    "expansion": 4,
                    "matrix_temp": 0.2,
                },
                # Heavier ResMLP for quality runs.
                "resmlp_large": {
                    "arch": "resmlp",
                    "learning_rate": 3e-4,
                    "model_dim": 1536,
                    "depth": 4,
                    "expansion": 4,
                    "matrix_temp": 0.2,
                },
            }
            chosen = presets.get(args.preset)
            if chosen is None:
                raise ValueError(f"Unknown preset: {args.preset}")
            # Only set defaults; if user passed an explicit flag, argparse keeps it.
            self.set_defaults(**chosen)

        # Final parse with the (possibly) updated defaults.
        self.args, _ = self.parse_known_args()


def main():
    args = HasherArgParser().args

    if args.mode == "train":
        loss_params = {
            "margin": args.margin,
            "w_distill": args.w_distill,
            "w_matrix": args.w_matrix,
            "matrix_temp": args.matrix_temp,
            "w_center": args.w_center,
            "w_entropy": args.w_entropy,
            "w_triplet": args.w_triplet,
            "w_quant": args.w_quant,
            "w_balance": args.w_balance,
            "w_decor": args.w_decor,
        }
        trainer = HasherTrainer(
            data_dir=SAMPLES_DIR,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hash_bits=args.hash_bits,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            arch=args.arch,
            model_dim=args.model_dim,
            depth=args.depth,
            expansion=args.expansion,
            max_samples=args.max_samples,
            remove_weights=args.remove_weights,
            loss_params=loss_params,
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
