from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


def build_backbone(encoder_cfg: Dict) -> Tuple[nn.Module, int]:
    backbone_type = encoder_cfg.get("type", "resnet50").lower()
    if backbone_type == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif backbone_type == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Unbekannter Backbone-Typ '{backbone_type}'.")
    out_dim = backbone.fc.in_features
    modules = list(backbone.children())[:-1]
    return nn.Sequential(*modules), out_dim


def build_embedding_head(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, out_dim),
    )


class CardEncoder(nn.Module):
    """
    Single-backbone Encoder fuer komplette Kartenbilder.
    """

    def __init__(self, cfg: Dict, num_classes: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        encoder_cfg = cfg.get("encoder", {})
        self.emb_dim = int(encoder_cfg.get("emb_dim", encoder_cfg.get("emb_full", 1024)))

        self.backbone, backbone_out_dim = build_backbone(encoder_cfg)
        self.head = build_embedding_head(backbone_out_dim, self.emb_dim)

        self.classifier = None
        if num_classes is not None and num_classes > 0:
            self.classifier = nn.Linear(self.emb_dim, num_classes)

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        """
        x: Tensor der Form (B, C, H, W) – Full-Card-Bilder
        """
        emb = self._encode_branch(self.backbone, self.head, x)
        emb = F.normalize(emb, p=2, dim=-1)

        if not return_logits:
            return emb

        logits = self.classifier(emb) if self.classifier is not None else None
        return emb, logits

    def _encode_branch(self, backbone: nn.Module, head: nn.Module, x: torch.Tensor) -> torch.Tensor:
        feat = backbone(x)
        emb = head(feat)
        return emb

    @torch.no_grad()
    def encode_normalized(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        emb = self.forward(x, return_logits=False)
        return emb


def build_encoder_model(cfg: Dict, num_classes: Optional[int] = None) -> CardEncoder:
    return CardEncoder(cfg, num_classes=num_classes)


def load_encoder(
    weights_path: str,
    cfg: Dict,
    num_classes: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> CardEncoder:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=device)
    metadata = {}
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        metadata = checkpoint
        state_dict = checkpoint["state_dict"]
        num_classes = num_classes or metadata.get("num_classes")
    model = CardEncoder(cfg, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_encoder: missing={missing}, unexpected={unexpected}")
    model.eval().to(device)
    model.card_ids = metadata.get("card_ids")
    return model


def save_encoder(model: CardEncoder, path: str, **metadata) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "num_classes": model.num_classes,
    }
    payload.update(metadata)
    torch.save(payload, path)
