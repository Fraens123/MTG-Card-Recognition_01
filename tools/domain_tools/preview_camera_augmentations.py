from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import sys
import yaml
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.augmentations import CameraLikeAugmentor
from src.core.image_ops import (
    crop_card_art,
    crop_set_symbol,
    get_full_art_crop_cfg,
    get_set_symbol_crop_cfg,
)


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
    """Resizes while preserving aspect ratio and letterboxes into tile."""
    target_w, target_h = size
    scale = min(target_w / img.width, target_h / img.height)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", size, color=(20, 20, 20))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(img_resized, offset)
    return canvas


def draw_label(tile: Image.Image, text: str) -> Image.Image:
    if not text:
        return tile
    tile = tile.copy()
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    padding = 4
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except AttributeError:
        w, h = font.getsize(text)
    box = [0, tile.height - h - padding * 2, tile.width, tile.height]
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((padding, tile.height - h - padding), text, font=font, fill=(255, 255, 255))
    return tile


def build_row(images: List[Image.Image], labels: List[str], tile_size: tuple[int, int]) -> Image.Image:
    width = tile_size[0] * len(images)
    height = tile_size[1]
    canvas = Image.new("RGB", (width, height), color=(30, 30, 30))
    for idx, img in enumerate(images):
        tile = resize_for_grid(img, tile_size)
        label = labels[idx] if labels else ""
        tile = draw_label(tile, label)
        canvas.paste(tile, (idx * tile_size[0], 0))
    return canvas


def save_grid(
    rows: List[Tuple[List[Image.Image], List[str]]],
    tile_size: tuple[int, int],
    output_path: Path,
) -> None:
    if not rows:
        return
    row_images = [build_row(images, labels, tile_size) for images, labels in rows]
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
    parser.add_argument("--no-symbols", action="store_true", help="Nur Artwork-Crops anzeigen, ohne Set-Symbole.")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="Optionaler Ordner, in dem alle verwendeten Crops (Art + Symbol) gespeichert werden.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    scryfall_dir = Path(config["data"]["scryfall_images"])
    camera_dir = Path(config["data"]["camera_images"])
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    augmentor_cfg = config.get("camera_augmentor", {})
    full_art_cfg = get_full_art_crop_cfg(config)
    set_symbol_cfg = get_set_symbol_crop_cfg(config)
    camera_augmentor = CameraLikeAugmentor(**augmentor_cfg)

    scryfall_samples = pick_images(scryfall_dir, args.num_images)
    if not scryfall_samples:
        raise SystemExit(f"Keine Scryfall-Bilder in {scryfall_dir} gefunden.")
    camera_refs = pick_images(camera_dir, args.camera_refs)
    ref_images = [Image.open(path).convert("RGB") for path in camera_refs]
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[List[Image.Image], List[str]]] = []
    show_symbols = not args.no_symbols
    for sample_idx, path in enumerate(scryfall_samples, start=1):
        base = Image.open(path).convert("RGB")
        aug_list = camera_augmentor.create_camera_like_augmentations(base, num_augmentations=args.per_image)
        # create_camera_like_augmentations liefert [original, aug1, aug2, ...]
        variants = [base] + aug_list[1 : args.per_image + 1]
        labels = ["base"] + [f"aug_{idx:02d}" for idx in range(1, len(variants))]
        if ref_images:
            variants += ref_images
            labels += [f"cam_{idx:02d}" for idx in range(1, len(ref_images) + 1)]
        art_row = [crop_card_art(img, full_art_cfg) for img in variants]
        rows.append((art_row, labels))
        symbol_row: List[Image.Image] = []
        if show_symbols and set_symbol_cfg:
            symbol_row = [crop_set_symbol(img, set_symbol_cfg) for img in variants]
            rows.append((symbol_row, labels))

        if dump_dir:
            sample_dir = dump_dir / f"{sample_idx:02d}_{path.stem}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for label, art_img in zip(labels, art_row):
                art_img.save(sample_dir / f"{label}_art.png")
            if symbol_row:
                for label, sym_img in zip(labels, symbol_row):
                    sym_img.save(sample_dir / f"{label}_symbol.png")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"camera_aug_preview_{timestamp}.png"
    save_grid(rows, tile_size=(224, 320), output_path=output_path)


if __name__ == "__main__":
    main()
