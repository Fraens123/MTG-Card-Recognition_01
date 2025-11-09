from typing import Optional, Tuple
import torch
from torch import nn
import torchvision.models as models


class Encoder(nn.Module):
    """
    ResNet50 backbone with a projection head for embeddings and
    an optional classifier head for CE supervision.
    """

    def __init__(self, embed_dim: int = 512, num_classes: Optional[int] = None, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.embedding = nn.Linear(512, embed_dim, bias=True)
        self.num_classes = num_classes
        self.classifier = nn.Linear(embed_dim, num_classes) if num_classes else None

    def forward(
        self, x: torch.Tensor, return_logits: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)
        emb = self.embedding(feat)
        if return_logits:
            logits = self.classifier(emb) if self.classifier is not None else None
            return emb, logits
        return emb

    @torch.no_grad()
    def encode_normalized(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward(x)
        emb = nn.functional.normalize(emb, p=2, dim=-1)
        return emb


def load_encoder(
    weights_path: str,
    embed_dim: int = 512,
    num_classes: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Encoder:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        num_classes = num_classes or checkpoint.get("num_classes")
    if num_classes is None and isinstance(state_dict, dict) and "classifier.weight" in state_dict:
        num_classes = state_dict["classifier.weight"].shape[0]
    model = Encoder(embed_dim=embed_dim, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model


def save_encoder(model: Encoder, path: str, **metadata):
    payload = {"state_dict": model.state_dict(), "num_classes": model.num_classes}
    payload.update(metadata)
    torch.save(payload, path)
