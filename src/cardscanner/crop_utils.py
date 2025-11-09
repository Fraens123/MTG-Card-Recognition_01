from __future__ import annotations

from typing import Optional

from PIL import Image
import yaml

_SYMBOL_CFG_CACHE: Optional[dict] = None


def load_symbol_crop_cfg() -> Optional[dict]:
    """
    Lädt (und cached) die Set-Symbol-Crop-Konfiguration aus config.yaml.
    """
    global _SYMBOL_CFG_CACHE
    if _SYMBOL_CFG_CACHE is not None:
        return _SYMBOL_CFG_CACHE
    try:
        with open("config.yaml", "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        _SYMBOL_CFG_CACHE = config.get("debug", {}).get("set_symbol_crop", None)
    except Exception as exc:
        print(f"[WARN] crop_utils: kann config.yaml nicht laden: {exc}")
        _SYMBOL_CFG_CACHE = None
    return _SYMBOL_CFG_CACHE


def crop_set_symbol(img: Image.Image, crop_cfg: Optional[dict] = None) -> Image.Image:
    """
    Schneidet das Set-Symbol aus dem Bild anhand relativer Koordinaten (0..1).
    """
    if crop_cfg is None:
        crop_cfg = load_symbol_crop_cfg()
    if not crop_cfg:
        return img

    x_min = float(crop_cfg.get("x_min", 0.7))
    y_min = float(crop_cfg.get("y_min", 0.3))
    x_max = float(crop_cfg.get("x_max", 0.95))
    y_max = float(crop_cfg.get("y_max", 0.6))

    w, h = img.size
    left = int(x_min * w)
    top = int(y_min * h)
    right = int(x_max * w)
    bottom = int(y_max * h)

    left = max(0, min(left, w - 1))
    right = max(left + 1, min(right, w))
    top = max(0, min(top, h - 1))
    bottom = max(top + 1, min(bottom, h))

    return img.crop((left, top, right, bottom))
