#!/usr/bin/env python3
"""
Vergleicht Embeddings zwischen Export- und Runtime-Pipeline an einem Beispielbild.
Nutzt dieselben Transforms wie export_embeddings.py bzw. recognize_cards.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch

# Sicherstellen, dass Repo-Root in sys.path liegt (erlaubt `import src...` auch bei direktem Aufruf).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_utils import load_config  # type: ignore
from src.core.embedding_utils import build_card_embedding  # type: ignore
from src.core.image_ops import (  # type: ignore
    build_resize_normalize_transform,
    crop_card_art,
    get_full_art_crop_cfg,
    resolve_resize_hw,
)
from src.core.model_builder import load_encoder  # type: ignore


def _crop_card_roi(img: Image.Image, roi_cfg: Optional[Dict]) -> Image.Image:
    """Roi-Crop wie in recognize_cards.crop_card_roi."""
    if not roi_cfg:
        return img
    w, h = img.size
    x0 = int(float(roi_cfg.get("x_min", 0.0)) * w)
    y0 = int(float(roi_cfg.get("y_min", 0.0)) * h)
    x1 = int(float(roi_cfg.get("x_max", 1.0)) * w)
    y1 = int(float(roi_cfg.get("y_max", 1.0)) * h)
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return img.crop((x0, y0, x1, y1))


def _to_hw(size_cfg: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if not size_cfg:
        return 320, 224
    width, height = size_cfg
    return int(height), int(width)


def compare_pipelines(config_path: str, image_path: str, model_path: Optional[str] = None) -> float:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("hardware", {}).get("use_cuda", True) else "cpu")

    export_cfg = cfg.get("embedding_export_runtime", {})
    default_model = export_cfg.get("model_path") or cfg.get("model", {}).get("path")
    model_path = model_path or default_model
    if not model_path:
        raise ValueError("Kein model_path gefunden (weder --model noch embedding_export_runtime.model_path).")
    model = load_encoder(model_path, cfg, device=device)
    model.eval()

    crop_cfg = get_full_art_crop_cfg(cfg)
    card_roi_cfg = cfg.get("camera", {}).get("card_roi")
    images_cfg = cfg.get("images", {})

    export_hw = _to_hw(tuple(images_cfg.get("full_card_size", [224, 320])))
    runtime_hw = resolve_resize_hw(cfg, cfg.get("paths", {}).get("scryfall_dir"))
    export_transform = build_resize_normalize_transform(export_hw)
    runtime_transform = build_resize_normalize_transform(runtime_hw)

    img = Image.open(image_path).convert("RGB")
    export_img = crop_card_art(img, crop_cfg)
    runtime_img = crop_card_art(_crop_card_roi(img, card_roi_cfg), crop_cfg)

    export_tensor = export_transform(export_img).unsqueeze(0).to(device)
    runtime_tensor = runtime_transform(runtime_img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb_export = build_card_embedding(model, export_tensor).cpu().numpy().flatten()
        emb_runtime = build_card_embedding(model, runtime_tensor).cpu().numpy().flatten()

    denom = (np.linalg.norm(emb_export) * np.linalg.norm(emb_runtime)) + 1e-8
    cosine = float(np.dot(emb_export, emb_runtime) / denom)

    print(f"[DEBUG] Export HW: {export_hw}, Runtime HW: {runtime_hw}")
    print(f"[DEBUG] Export crop size: {export_img.size}, Runtime crop size nach ROI: {runtime_img.size}")
    print(f"[DEBUG] Cosine Similarity (Export vs. Runtime): {cosine:.6f}")
    return cosine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vergleicht Embeddings zwischen Export- und Runtime-Pipeline.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Pfad zur YAML-Config (z.B. config.train20k.yaml).")
    parser.add_argument("--image", "-i", type=str, required=True, help="Pfad zu einem Scryfall-Bild.")
    parser.add_argument("--model", "-m", type=str, default=None, help="Optionaler Pfad zum Encoder-Gewicht.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_pipelines(args.config, args.image, model_path=args.model)


if __name__ == "__main__":
    main()
