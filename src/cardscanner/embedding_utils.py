from __future__ import annotations

import torch
import torch.nn.functional as F


def build_card_embedding(model: torch.nn.Module, full_tensor: torch.Tensor, crop_tensor: torch.Tensor) -> torch.Tensor:
    """
    Erzeugt ein normalisiertes Karten-Embedding bestehend aus Full- und Crop-Vektor.
    Wichtig: Dieses Schema wird für Export und Query identisch verwendet.
    """
    emb_full = model(full_tensor)
    emb_full = F.normalize(emb_full, p=2, dim=-1)

    emb_crop = model(crop_tensor)
    emb_crop = F.normalize(emb_crop, p=2, dim=-1)

    combined = torch.cat([emb_full, emb_crop], dim=-1)
    combined = F.normalize(combined, p=2, dim=-1)
    return combined
