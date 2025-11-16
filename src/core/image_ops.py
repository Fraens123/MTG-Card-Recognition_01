from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image
import torchvision.transforms as T

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def detect_image_size(images_dir: str, max_dim: int = 400) -> Tuple[int, int]:
    """Detects a reasonable resize target based on available images."""
    images_path = Path(images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    for entry in images_path.iterdir():
        if entry.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(entry) as img:
            width, height = img.size
        if max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            width = int(width * scale)
            height = int(height * scale)
        width = (width // 8) * 8 or 224
        height = (height // 8) * 8 or 320
        return width, height

    return 224, 320


def build_resize_normalize_transform(resize_hw: Tuple[int, int]) -> T.Compose:
    """Common resize + normalize pipeline."""
    return T.Compose(
        [
            T.Resize(resize_hw, antialias=True),
            T.ToTensor(),
            T.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )


def _ensure_hw_tuple(size_cfg: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if not size_cfg:
        return 320, 224
    width, height = size_cfg
    return int(height), int(width)


def get_full_art_crop_cfg(config: Dict) -> Optional[Dict]:
    images_cfg = config.get("images", {})
    crop_cfg = images_cfg.get("full_crop")
    if crop_cfg:
        return crop_cfg
    return config.get("debug", {}).get("full_art_crop")


def resolve_resize_hw(config: Dict, sample_dir: Optional[str] = None) -> Tuple[int, int]:
    """
    Liefert die Zielgr��e (H, W) f�r Vollkarten. Priorisiert config.yaml,
    f�llt aber optional per auto-detect, falls keine Werte vorhanden sind.
    """
    images_cfg = config.get("images", {})
    full_size = images_cfg.get("full_card_size")
    if full_size:
        return _ensure_hw_tuple(tuple(full_size))
    if sample_dir:
        width, height = detect_image_size(sample_dir)
        return height, width
    return 320, 224


def _crop_region(img: Image.Image, crop_cfg: Optional[Dict]) -> Image.Image:
    if not crop_cfg:
        return img

    w, h = img.size
    x0 = int(crop_cfg.get("x_min", 0.0) * w)
    y0 = int(crop_cfg.get("y_min", 0.0) * h)
    x1 = int(crop_cfg.get("x_max", 1.0) * w)
    y1 = int(crop_cfg.get("y_max", 1.0) * h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return img

    crop = img.crop((x0, y0, x1, y1))
    target_w = int(crop_cfg.get("target_width", w))
    target_h = int(crop_cfg.get("target_height", h))
    keep_aspect = crop_cfg.get("keep_aspect", True)

    if not keep_aspect:
        return crop.resize((target_w, target_h), Image.BILINEAR)

    crop_w, crop_h = crop.size
    if crop_w == 0 or crop_h == 0:
        return crop

    scale = min(target_w / crop_w, target_h / crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    resized = crop.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target_w, target_h), color=(0, 0, 0))
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def crop_card_art(img: Image.Image, crop_cfg: Optional[Dict]) -> Image.Image:
    return _crop_region(img, crop_cfg)
