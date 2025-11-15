from __future__ import annotations

import torch


def build_card_embedding(
    model: torch.nn.Module,
    full_tensor: torch.Tensor,
    crop_tensor: torch.Tensor,
) -> torch.Tensor:
    """Erzeugt ein gemeinsames, L2-normalisiertes Karten-Embedding (Full + Symbol)."""
    if full_tensor.ndim == 3:
        full_tensor = full_tensor.unsqueeze(0)
    if crop_tensor.ndim == 3:
        crop_tensor = crop_tensor.unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_normalized(full_tensor, crop_tensor)
    return embedding.squeeze(0)
