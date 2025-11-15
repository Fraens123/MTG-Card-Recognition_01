from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


def build_backbone(cfg: Dict) -> Tuple[nn.Module, int]:
    encoder_cfg = cfg.get("encoder", {})
    backbone_type = encoder_cfg.get("type", "resnet18").lower()
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
    Encoder mit getrennten Backbones fuer Full-, Symbol- und Namenspfad.
    Die resultierenden Embeddings werden konkateniert und L2-normalisiert.
    """

    def __init__(self, cfg: Dict, num_classes: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        self.backbone_full, full_out_dim = build_backbone(cfg)
        self.backbone_symbol, symbol_out_dim = build_backbone(cfg)
        self.backbone_name, name_out_dim = build_backbone(cfg)

        encoder_cfg = cfg.get("encoder", {})
        self.emb_full_dim = int(encoder_cfg.get("emb_full", 512))
        self.emb_symbol_dim = int(encoder_cfg.get("emb_symbol", 512))
        self.emb_name_dim = int(encoder_cfg.get("emb_name", 512))

        self.head_full = build_embedding_head(full_out_dim, self.emb_full_dim)
        self.head_symbol = build_embedding_head(symbol_out_dim, self.emb_symbol_dim)
        self.head_name = build_embedding_head(name_out_dim, self.emb_name_dim)

        if num_classes:
            self.classifier_full = nn.Linear(self.emb_full_dim, num_classes)
            self.classifier_symbol = nn.Linear(self.emb_symbol_dim, num_classes)
            self.classifier_name = nn.Linear(self.emb_name_dim, num_classes)
        else:
            self.classifier_full = None
            self.classifier_symbol = None
            self.classifier_name = None

    def forward(
        self,
        x_full: torch.Tensor,
        x_symbol: torch.Tensor,
        x_name: torch.Tensor,
        return_logits: bool = False,
    ):
        emb_full = self._encode_branch(self.backbone_full, self.head_full, x_full)
        emb_symbol = self._encode_branch(self.backbone_symbol, self.head_symbol, x_symbol)
        emb_name = self._encode_branch(self.backbone_name, self.head_name, x_name)

        combined = torch.cat([emb_full, emb_symbol, emb_name], dim=1)
        combined = F.normalize(combined, p=2, dim=-1)

        if return_logits:
            logits_full = self.classifier_full(emb_full) if self.classifier_full is not None else None
            logits_symbol = self.classifier_symbol(emb_symbol) if self.classifier_symbol is not None else None
            logits_name = self.classifier_name(emb_name) if self.classifier_name is not None else None
            return combined, (logits_full, logits_symbol, logits_name)
        return combined

    def _encode_branch(self, backbone: nn.Module, head: nn.Module, x: torch.Tensor) -> torch.Tensor:
        feat = backbone(x)
        emb = head(feat)
        return F.normalize(emb, p=2, dim=-1)

    @torch.no_grad()
    def encode_normalized(
        self,
        x_full: torch.Tensor,
        x_symbol: Optional[torch.Tensor] = None,
        x_name: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_symbol is None or x_name is None:
            if isinstance(x_full, (tuple, list)) and len(x_full) == 3:
                x_full, x_symbol, x_name = x_full
            else:
                raise ValueError("CardEncoder.encode_normalized erwartet full-, symbol- und name-Tensor.")
        embedding = self.forward(x_full, x_symbol, x_name)
        return embedding


def build_card_encoder(cfg: Dict, num_classes: Optional[int] = None) -> CardEncoder:
    return CardEncoder(cfg, num_classes=num_classes)


def build_encoder_model(cfg: Dict, num_classes: Optional[int] = None) -> CardEncoder:
    """Rueckwaertskompatibler Alias fuer build_card_encoder."""
    return build_card_encoder(cfg, num_classes=num_classes)


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
