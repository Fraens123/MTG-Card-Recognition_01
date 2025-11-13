from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import List

import sys
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cardscanner.augment_cards import CameraLikeAugmentor


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def pick_images(folder: Path, count: int) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()]
    if len(files) <= count:
        return files
    rng = random.Random(1337)
    return rng.sample(files, count)


def resize_for_grid(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.BILINEAR)


def build_row(images: List[Image.Image], tile_size: tuple[int, int]) -> Image.Image:
    width = tile_size[0] * len(images)
    height = tile_size[1]
    canvas = Image.new("RGB", (width, height), color=(30, 30, 30))
    for idx, img in enumerate(images):
        canvas.paste(resize_for_grid(img, tile_size), (idx * tile_size[0], 0))
    return canvas


def save_grid(rows: List[List[Image.Image]], tile_size: tuple[int, int], output_path: Path) -> None:
    if not rows:
        return
    row_images = [build_row(row, tile_size) for row in rows]
    width = max(img.width for img in row_images)
    height = sum(img.height for img in row_images)
    grid = Image.new("RGB", (width, height), color=(20, 20, 20))
    y = 0
    for row in row_images:
        grid.paste(row, (0, y))
        y += row.height
    grid.save(output_path)
    print(f"[INFO] Augmentierungs-Preview gespeichert: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualisiert CameraLikeAugmentor-Ausgaben.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--num-images", type=int, default=5, help="Anzahl zufaelliger Scryfall-Bilder")
    parser.add_argument("--per-image", type=int, default=6, help="Augmentierungen pro Bild")
    parser.add_argument("--camera-refs", type=int, default=2, help="Anzahl Pi-Cam-Referenzen pro Reihe")
    parser.add_argument("--output", default="debug/domain_preview", help="Zielordner fuer Previews")
    args = parser.parse_args()

    config = load_config(args.config)
    scryfall_dir = Path(config["data"]["scryfall_images"])
    camera_dir = Path(config["data"]["camera_images"])
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    augmentor_cfg = config.get("camera_augmentor", {})
    camera_augmentor = CameraLikeAugmentor(**augmentor_cfg)

    scryfall_samples = pick_images(scryfall_dir, args.num_images)
    if not scryfall_samples:
        raise SystemExit(f"Keine Scryfall-Bilder in {scryfall_dir} gefunden.")
    camera_refs = pick_images(camera_dir, args.camera_refs)
    ref_images = [Image.open(path).convert("RGB") for path in camera_refs]

    rows: List[List[Image.Image]] = []
    for path in scryfall_samples:
        base = Image.open(path).convert("RGB")
        aug_list = camera_augmentor.create_camera_like_augmentations(base, num_augmentations=args.per_image)
        # create_camera_like_augmentations liefert [original, aug1, aug2, ...]
        variants = aug_list[1 : args.per_image + 1]
        row_images = [base] + variants
        if ref_images:
            row_images += ref_images
        rows.append(row_images)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"camera_aug_preview_{timestamp}.png"
    save_grid(rows, tile_size=(224, 320), output_path=output_path)


if __name__ == "__main__":
    main()
