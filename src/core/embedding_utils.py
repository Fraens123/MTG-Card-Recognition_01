from __future__ import annotations

import torch


def build_card_embedding(model: torch.nn.Module, full_tensor: torch.Tensor) -> torch.Tensor:
    """
    Baut ein L2-normalisiertes Karten-Embedding aus einem Full-Card-Bild.
    """
    if full_tensor.ndim == 3:
        full_tensor = full_tensor.unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_normalized(full_tensor)
    return embedding.squeeze(0)
