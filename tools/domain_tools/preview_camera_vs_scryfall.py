#!/usr/bin/env python3
"""
Vergleicht Pi-Cam-ROI/Crops mit den entsprechenden Scryfall-Crops.

Speichert pro Kamera-Bild ein Preview mit:
- Originalfoto inkl. eingezeichneter ROI
- ROI-Ausschnitt (normierte Karte)
- Artwork-Crop dieses Kamera-Bildes
- (falls vorhanden) Original Scryfall-Bild + dessen Artwork-Crop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.image_ops import crop_card_art, get_full_art_crop_cfg


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def crop_card_roi(img: Image.Image, roi_cfg: Optional[Dict]) -> Image.Image:
    if not roi_cfg:
        return img.copy()
    w, h = img.size
    try:
        x_min = float(roi_cfg.get("x_min", 0.0))
        y_min = float(roi_cfg.get("y_min", 0.0))
        x_max = float(roi_cfg.get("x_max", 1.0))
        y_max = float(roi_cfg.get("y_max", 1.0))
    except Exception:
        return img.copy()
    x0 = int(x_min * w)
    y0 = int(y_min * h)
    x1 = int(x_max * w)
    y1 = int(y_max * h)
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return img.crop((x0, y0, x1, y1))


def draw_overlay(
    img: Image.Image,
    roi_cfg: Optional[Dict],
    name_box: Optional[Tuple[int, int, int, int]] = None,
    collector_box: Optional[Tuple[int, int, int, int]] = None,
    setid_box: Optional[Tuple[int, int, int, int]] = None,
) -> Image.Image:
    """Zeichnet Karten-ROI sowie optional OCR-Boxen für Name / Collector / SetID."""
    overlay = img.copy()
    w, h = overlay.size
    draw = ImageDraw.Draw(overlay)

    # Karten ROI
    if roi_cfg:
        try:
            x0 = int(float(roi_cfg.get("x_min", 0.0)) * w)
            y0 = int(float(roi_cfg.get("y_min", 0.0)) * h)
            x1 = int(float(roi_cfg.get("x_max", 1.0)) * w)
            y1 = int(float(roi_cfg.get("y_max", 1.0)) * h)
            draw.rectangle((x0, y0, x1, y1), outline="red", width=max(2, int(min(w, h) * 0.005)))
        except Exception:
            pass

    # Name Box (gelb)
    if name_box:
        nx, ny, nw, nh = name_box
        draw.rectangle((nx, ny, nx + nw, ny + nh), outline="yellow", width=3)
    # Collector Box (orange)
    if collector_box:
        cx, cy, cw, ch = collector_box
        draw.rectangle((cx, cy, cx + cw, cy + ch), outline="orange", width=3)
    # SetID Box (cyan)
    if setid_box:
        sx, sy, sw, sh = setid_box
        draw.rectangle((sx, sy, sx + sw, sy + sh), outline="cyan", width=3)

    return overlay


def normalize_name(stem: str) -> str:
    return "".join(ch for ch in stem.lower() if ch.isalnum())


def build_scryfall_index(scryfall_dir: Path) -> Dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png"}
    index: Dict[str, Path] = {}
    for path in scryfall_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        index[normalize_name(path.stem)] = path
    return index


def find_matching_scryfall(camera_path: Path, index: Dict[str, Path]) -> Optional[Path]:
    cam_key = normalize_name(camera_path.stem)
    if not cam_key:
        return None
    # exakter Treffer
    if cam_key in index:
        return index[cam_key]
    # substring-suche
    for key, path in index.items():
        if cam_key in key or key in cam_key:
            return path
    return None


def prepare_tile(img: Image.Image, label: str, width: int) -> Image.Image:
    if img.width != width:
        scale = width / img.width
        img = img.resize((width, max(1, int(img.height * scale))), Image.BILINEAR)
    label_height = 36
    canvas = Image.new("RGB", (width, img.height + label_height), color=(15, 15, 15))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = font.getsize(label)
    draw.rectangle((0, img.height, width, img.height + label_height), fill=(0, 0, 0))
    draw.text(
        ((width - text_w) // 2, img.height + (label_height - text_h) // 2),
        label,
        fill=(255, 255, 255),
        font=font,
    )
    return canvas


def stack_sections(sections: List[List[Image.Image]], gap: int = 12) -> Image.Image:
    width = max(img.width for section in sections for img in section)
    section_gap = gap * 2
    total_height = -section_gap
    composed_sections: List[Image.Image] = []
    for section in sections:
        height = sum(img.height for img in section) + gap * (len(section) - 1)
        canvas = Image.new("RGB", (width, height), color=(10, 10, 10))
        y = 0
        for img in section:
            canvas.paste(img, ((width - img.width) // 2, y))
            y += img.height + gap
        composed_sections.append(canvas)
        total_height += canvas.height + section_gap
    total_height += section_gap

    preview = Image.new("RGB", (width, total_height), color=(5, 5, 5))
    y = 0
    for section in composed_sections:
        preview.paste(section, (0, y))
        y += section.height + section_gap
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Vergleich Pi-Cam Crops vs. Scryfall Crops (inkl. OCR-Boxen)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--camera-dir", default=None, help="Override fuer paths.camera_dir")
    parser.add_argument("--scryfall-dir", default=None, help="Override fuer paths.scryfall_dir")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--output-dir", default="debug/camera_vs_scryfall")
    # OCR Box Parameter
    parser.add_argument("--name-x", type=int, default=100)
    parser.add_argument("--name-y", type=int, default=50)
    parser.add_argument("--name-w", type=int, default=2750)
    parser.add_argument("--name-h", type=int, default=200)
    parser.add_argument("--collector-x", type=int, default=100)
    parser.add_argument("--collector-y", type=int, default=3600)
    parser.add_argument("--collector-w", type=int, default=500)
    parser.add_argument("--collector-h", type=int, default=80)
    parser.add_argument("--setid-x", type=int, default=100)
    parser.add_argument("--setid-y", type=int, default=3700)
    parser.add_argument("--setid-w", type=int, default=500)
    parser.add_argument("--setid-h", type=int, default=80)
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config.get("paths", {})
    camera_dir = Path(args.camera_dir or paths_cfg.get("camera_dir", "./data/camera_images"))
    scryfall_dir = Path(args.scryfall_dir or paths_cfg.get("scryfall_dir", "./data/scryfall_images"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_cfg = config.get("camera", {}).get("card_roi", None)
    art_cfg = get_full_art_crop_cfg(config)

    cam_files = sorted([p for p in camera_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not cam_files:
        raise SystemExit(f"Keine Kamera-Bilder in {camera_dir}")

    scryfall_index = build_scryfall_index(scryfall_dir)
    scryfall_files = sorted(scryfall_index.values())
    if not scryfall_files:
        raise SystemExit(f"Keine Scryfall-Bilder in {scryfall_dir}")

    for idx, cam_path in enumerate(cam_files[: args.max_images], start=1):
        with Image.open(cam_path) as cam_img_raw:
            cam_img = cam_img_raw.convert("RGB")
        name_box = (args.name_x, args.name_y, args.name_w, args.name_h)
        collector_box = (args.collector_x, args.collector_y, args.collector_w, args.collector_h)
        setid_box = (args.setid_x, args.setid_y, args.setid_w, args.setid_h)
        overlay = draw_overlay(cam_img, roi_cfg, name_box, collector_box, setid_box)
        card = crop_card_roi(cam_img, roi_cfg)
        cam_art = crop_card_art(card, art_cfg)

        tiles_camera = [
            prepare_tile(overlay, "Pi-Cam Original + ROI + OCR", 900),
            prepare_tile(card, "Pi-Cam ROI-Karte", 900),
            prepare_tile(cam_art, "Pi-Cam Artwork-Crop", 900),
        ]

        tiles_scryfall: List[Image.Image] = []
        scry_path = find_matching_scryfall(cam_path, scryfall_index)
        if not scry_path:
            scry_path = scryfall_files[(idx - 1) % len(scryfall_files)]
            fallback_label = "(Fallback)"
        else:
            fallback_label = ""
        if scry_path:
            with Image.open(scry_path) as scry_img_raw:
                scry_img = scry_img_raw.convert("RGB")
            scry_art = crop_card_art(scry_img, art_cfg)
            scry_overlay = draw_overlay(scry_img, None, name_box, collector_box, setid_box)
            tiles_scryfall = [
                prepare_tile(scry_overlay, f"Scryfall Original+OCR {fallback_label}".strip(), 900),
                prepare_tile(scry_art, "Scryfall Artwork-Crop", 900),
            ]

        preview = stack_sections([tiles_camera, tiles_scryfall])
        out_path = output_dir / f"{idx:02d}_{cam_path.stem}_preview.png"
        preview.save(out_path)
        print(f"[PREVIEW] {out_path}")


if __name__ == "__main__":
    main()
