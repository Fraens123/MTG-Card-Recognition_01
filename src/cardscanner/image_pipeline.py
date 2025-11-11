from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image
import torchvision.transforms as T

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def detect_image_size(images_dir: str, max_dim: int = 400) -> Tuple[int, int]:
    """
    Ermittelt eine sinnvolle Zielgröße basierend auf den vorhandenen Bildern.
    Skaliert das erste gefundene Bild auf max_dim und rundet auf Vielfache von 8.
    """
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

    # Fallback, falls kein Bild gefunden wurde
    return 224, 320


def build_resize_normalize_transform(resize_hw: Tuple[int, int]) -> T.Compose:
    """
    Einfache Pipeline: Resize -> ToTensor -> Normalize.
    Diese Funktion wird von Training, Export und Query gemeinsam genutzt.
    """
    return T.Compose(
        [
            T.Resize(resize_hw, antialias=True),
            T.ToTensor(),
            T.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )


def get_set_symbol_crop_cfg(config: Dict) -> Optional[Dict]:
    """Extrahiert die Set-Symbol-Crop-Konfiguration aus config.yaml."""
    return config.get("debug", {}).get("set_symbol_crop")


def resolve_resize_hw(config: Dict, sample_dir: Optional[str] = None) -> Tuple[int, int]:
    """
    Liefert (height, width) für die Resize-Pipeline.
    Nutzt training.auto_detect_size, andernfalls target_width/height.
    """
    training_cfg = config.get("training", {})
    if training_cfg.get("auto_detect_size"):
        if not sample_dir:
            raise ValueError("auto_detect_size requires a sample directory path")
        width, height = detect_image_size(sample_dir)
    else:
        width = training_cfg.get("target_width", 224)
        height = training_cfg.get("target_height", 320)
    return height, width
