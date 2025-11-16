from __future__ import annotations

from typing import List

import numpy as np
import torch


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def compute_centroid(vectors: List[np.ndarray]) -> np.ndarray:
    c = np.mean(vectors, axis=0)
    return l2_normalize(c)


def build_card_embedding(model: torch.nn.Module, full_tensor: torch.Tensor) -> torch.Tensor:
    """
    Baut ein L2-normalisiertes Karten-Embedding aus einem Full-Card-Bild.
    """
    if full_tensor.ndim == 3:
        full_tensor = full_tensor.unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_normalized(full_tensor)
    return embedding.squeeze(0)


def build_card_embedding_batch(model: torch.nn.Module, full_batch: torch.Tensor) -> torch.Tensor:
    """
    Batch-Version von build_card_embedding. Erwartet [B, C, H, W] und liefert [B, D].
    """
    assert full_batch.ndim == 4, "full_batch muss 4D sein (B, C, H, W)."
    with torch.no_grad():
        embeddings = model.encode_normalized(full_batch)
    return embeddings
