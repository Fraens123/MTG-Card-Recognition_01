#!/usr/bin/env python3
"""
Preview & Tuning Tool für OCR-Crops (nur Kamera-Bilder).

Erstellt für eine zufällige Auswahl von Kamera-Bildern:
- Overlay des Originalbilds mit eingezeichneten OCR-Regionen (Name / Collector / SetID) optional ROI
- Einzelne Crop-Dateien
- Ein zusammengesetztes Preview (Original + Overlays + Crops)

Verwendung:
    python tools/domain_tools/preview_ocr_crops.py \
        --config config.yaml \
        --num-camera 8 \
        --output-dir debug/ocr_tuning \
        --name-x-start 100 --name-x-end 2850 --name-y 50 --name-h 200 \
        --collector-x 100 --collector-y 3600 --collector-w 500 --collector-h 80 \
        --setid-x 100 --setid-y 3700 --setid-w 500 --setid-h 80

Alle Parameter sind optional; Standardwerte entsprechen den aktuellen Konstanten.

Ziel: Visuelle Kontrolle und Feineinstellung der Crop-Koordinaten.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List

import yaml
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default-Konstanten (Name jetzt x,y,w,h; rückwärtskompatibel)
DEFAULT_NAME_X = 100
DEFAULT_NAME_Y = 50
DEFAULT_NAME_W = 2750  # 100..2850
DEFAULT_NAME_H = 200
DEFAULT_COLLECTOR = (100, 2700, 500, 80)  # x,y,w,h (angepasst: vorher außerhalb Bildhöhe)
DEFAULT_SETID = (100, 2800, 500, 80)      # x,y,w,h (angepasst: vorher außerhalb Bildhöhe)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]


def draw_ocr_overlay(img: Image.Image, args, roi_cfg: dict | None = None, show_roi: bool = False) -> Image.Image:
    """Erzeugt Overlay mit OCR-Crop-Boxen und optional Karten-ROI (rot).

    Ergänzungen:
    - Clamping der Y-Koordinaten für Collector / SetID, falls außerhalb der Bildhöhe
    - Text-Labels direkt auf dem Overlay für schnellere visuelle Zuordnung
    """
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    W, H = overlay.size
    font = ImageFont.load_default()

    # Optional ROI Box (gesamte Karte im Kamera-Bild)
    if show_roi and roi_cfg:
        try:
            x0 = int(float(roi_cfg.get("x_min", 0.0)) * W)
            y0 = int(float(roi_cfg.get("y_min", 0.0)) * H)
            x1 = int(float(roi_cfg.get("x_max", 1.0)) * W)
            y1 = int(float(roi_cfg.get("y_max", 1.0)) * H)
            draw.rectangle((x0, y0, x1, y1), outline="red", width=3)
            draw.text((x0 + 6, y0 + 6), "ROI", fill="red", font=font)
        except Exception:
            pass

    # Name Region (Gelb) neues Schema
    name_y0 = max(0, min(args.name_y, H - args.name_h))
    name_y1 = name_y0 + args.name_h
    draw.rectangle(
        (
            args.name_x,
            name_y0,
            args.name_x + args.name_w,
            name_y1,
        ),
        outline="yellow",
        width=4,
    )
    draw.text((args.name_x + 6, name_y0 + 6), "NAME", fill="yellow", font=font)

    # Collector (Orange) mit Clamping
    coll_y0 = max(0, min(args.collector_y, H - args.collector_h))
    coll_y1 = coll_y0 + args.collector_h
    draw.rectangle(
        (
            args.collector_x,
            coll_y0,
            args.collector_x + args.collector_w,
            coll_y1,
        ),
        outline="orange",
        width=3,
    )
    draw.text((args.collector_x + 6, coll_y0 + 6), "COLLECTOR", fill="orange", font=font)

    # SetID (Cyan) mit Clamping
    set_y0 = max(0, min(args.setid_y, H - args.setid_h))
    set_y1 = set_y0 + args.setid_h
    draw.rectangle(
        (
            args.setid_x,
            set_y0,
            args.setid_x + args.setid_w,
            set_y1,
        ),
        outline="cyan",
        width=3,
    )
    draw.text((args.setid_x + 6, set_y0 + 6), "SETID", fill="cyan", font=font)
    return overlay


def crop_region(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = box
    return img.crop((x0, y0, x1, y1))


def safe_crop(img: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    W, H = img.size
    x0 = max(0, min(x, W - 1))
    y0 = max(0, min(y, H - 1))
    x1 = max(x0 + 1, min(x + w, W))
    y1 = max(y0 + 1, min(y + h, H))
    return img.crop((x0, y0, x1, y1))


def label_image(img: Image.Image, label: str, width: int) -> Image.Image:
    if img.width != width:
        scale = width / img.width
        img = img.resize((width, max(1, int(img.height * scale))), Image.BILINEAR)
    label_h = 40
    canvas = Image.new("RGB", (width, img.height + label_h), color=(20, 20, 20))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, img.height, width, img.height + label_h), fill=(0, 0, 0))
    draw.text((10, img.height + 10), label, fill=(255, 255, 255), font=font)
    return canvas


def compose_preview(tiles: List[Image.Image], cols: int = 3, gap: int = 12) -> Image.Image:
    if not tiles:
        return Image.new("RGB", (800, 200), color=(30, 30, 30))
    width = max(t.width for t in tiles)
    # Resize all to same width for clean grid
    tiles = [t if t.width == width else t.resize((width, int(t.height * (width / t.width))), Image.BILINEAR) for t in tiles]
    rows = (len(tiles) + cols - 1) // cols
    h_per_row = [max(tiles[r*cols + c].height for c in range(cols) if r*cols + c < len(tiles)) for r in range(rows)]
    total_h = sum(h_per_row) + gap * (rows + 1)
    total_w = cols * width + gap * (cols + 1)
    canvas = Image.new("RGB", (total_w, total_h), color=(15, 15, 15))
    y = gap
    idx = 0
    for r in range(rows):
        x = gap
        for c in range(cols):
            if idx >= len(tiles):
                break
            canvas.paste(tiles[idx], (x, y))
            x += width + gap
            idx += 1
        y += h_per_row[r] + gap
    return canvas


def process_set(label: str, paths: List[Path], args, out_dir: Path, roi_cfg: dict | None, show_roi: bool) -> None:
    for i, path in enumerate(paths, start=1):
        try:
            with Image.open(path) as raw:
                img = raw.convert("RGB")
            overlay = draw_ocr_overlay(img, args, roi_cfg=roi_cfg, show_roi=show_roi)
            name_crop = safe_crop(img, args.name_x, args.name_y, args.name_w, args.name_h)
            collector_crop = safe_crop(img, args.collector_x, args.collector_y, args.collector_w, args.collector_h)
            setid_crop = safe_crop(img, args.setid_x, args.setid_y, args.setid_w, args.setid_h)

            # Speichern der Einzel-Crops
            base = f"{label}_{i:02d}_{path.stem}"
            name_crop.save(out_dir / f"{base}_name.png")
            collector_crop.save(out_dir / f"{base}_collector.png")
            setid_crop.save(out_dir / f"{base}_setid.png")
            overlay.save(out_dir / f"{base}_overlay.png")

            legend = "Overlay (Gelb=Name / Orange=Collector / Cyan=SetID" + (" / Rot=ROI" if show_roi and roi_cfg else "") + ")"
            tiles = [
                label_image(img, f"{label} Original", 600),
                label_image(overlay, legend, 600),
                label_image(name_crop, "Name-Crop", 600),
                label_image(collector_crop, "Collector-Crop", 600),
                label_image(setid_crop, "SetID-Crop", 600),
            ]
            preview = compose_preview(tiles, cols=3)
            preview.save(out_dir / f"{base}_preview.png")
            print(f"[PREVIEW] {base}")
        except Exception as e:
            print(f"[ERROR] {path}: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview & Tuning für OCR-Crops")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--camera-dir", default=None)
    p.add_argument("--num-camera", type=int, default=6)
    p.add_argument("--output-dir", default="debug/ocr_tuning")
    # Name Crop Overrides (neues Schema)
    p.add_argument("--name-x", type=int, default=DEFAULT_NAME_X)
    p.add_argument("--name-y", type=int, default=DEFAULT_NAME_Y)
    p.add_argument("--name-w", type=int, default=DEFAULT_NAME_W)
    p.add_argument("--name-h", type=int, default=DEFAULT_NAME_H)
    # Collector Crop Overrides
    p.add_argument("--collector-x", type=int, default=DEFAULT_COLLECTOR[0])
    p.add_argument("--collector-y", type=int, default=DEFAULT_COLLECTOR[1])
    p.add_argument("--collector-w", type=int, default=DEFAULT_COLLECTOR[2])
    p.add_argument("--collector-h", type=int, default=DEFAULT_COLLECTOR[3])
    # SetID Crop Overrides
    p.add_argument("--setid-x", type=int, default=DEFAULT_SETID[0])
    p.add_argument("--setid-y", type=int, default=DEFAULT_SETID[1])
    p.add_argument("--setid-w", type=int, default=DEFAULT_SETID[2])
    p.add_argument("--setid-h", type=int, default=DEFAULT_SETID[3])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show-roi", action="store_true", help="Zeige zusätzlich Karten-ROI (rot)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Falls config ocr_crops enthält: Werte als Default setzen (nur wenn Nutzer nichts überschreibt)
    crops_cfg = config.get("ocr_crops", {})
    name_cfg = crops_cfg.get("name", {})
    collector_cfg = crops_cfg.get("collector", {})
    setid_cfg = crops_cfg.get("setid", {})

    # Helper um festzustellen ob Argument noch Default ist (wir nehmen an Default == Konstanten oben)
    def is_default(val, default):
        return val == default

    # Name (unterstützt altes und neues Schema)
    if name_cfg:
        if "x" in name_cfg and is_default(args.name_x, DEFAULT_NAME_X):
            args.name_x = int(name_cfg["x"])
        if "y" in name_cfg and is_default(args.name_y, DEFAULT_NAME_Y):
            args.name_y = int(name_cfg["y"])
        if "w" in name_cfg and is_default(args.name_w, DEFAULT_NAME_W):
            args.name_w = int(name_cfg["w"])
        if "h" in name_cfg and is_default(args.name_h, DEFAULT_NAME_H):
            args.name_h = int(name_cfg["h"])
        # Altes Schema fallback (x_start/x_end/height)
        if "x_start" in name_cfg and "x_end" in name_cfg and is_default(args.name_w, DEFAULT_NAME_W):
            args.name_w = int(name_cfg["x_end"]) - int(name_cfg["x_start"])
        if "height" in name_cfg and is_default(args.name_h, DEFAULT_NAME_H) and "h" not in name_cfg:
            args.name_h = int(name_cfg["height"])
        if "x_start" in name_cfg and is_default(args.name_x, DEFAULT_NAME_X) and "x" not in name_cfg:
            args.name_x = int(name_cfg["x_start"])

    # Collector
    if collector_cfg:
        if is_default(args.collector_x, DEFAULT_COLLECTOR[0]) and "x" in collector_cfg:
            args.collector_x = int(collector_cfg["x"])
        if is_default(args.collector_y, DEFAULT_COLLECTOR[1]) and "y" in collector_cfg:
            args.collector_y = int(collector_cfg["y"])
        if is_default(args.collector_w, DEFAULT_COLLECTOR[2]) and "w" in collector_cfg:
            args.collector_w = int(collector_cfg["w"])
        if is_default(args.collector_h, DEFAULT_COLLECTOR[3]) and "h" in collector_cfg:
            args.collector_h = int(collector_cfg["h"])

    # SetID
    if setid_cfg:
        if is_default(args.setid_x, DEFAULT_SETID[0]) and "x" in setid_cfg:
            args.setid_x = int(setid_cfg["x"])
        if is_default(args.setid_y, DEFAULT_SETID[1]) and "y" in setid_cfg:
            args.setid_y = int(setid_cfg["y"])
        if is_default(args.setid_w, DEFAULT_SETID[2]) and "w" in setid_cfg:
            args.setid_w = int(setid_cfg["w"])
        if is_default(args.setid_h, DEFAULT_SETID[3]) and "h" in setid_cfg:
            args.setid_h = int(setid_cfg["h"])

    paths_cfg = config.get("paths", {})
    camera_dir = Path(args.camera_dir or paths_cfg.get("camera_dir", "./data/camera_images"))
    if not camera_dir.exists():
        raise SystemExit(f"Camera-Verzeichnis fehlt: {camera_dir}")

    cam_imgs = list_images(camera_dir)
    if not cam_imgs:
        raise SystemExit(f"Keine Kamera-Bilder gefunden in {camera_dir}")

    random.seed(args.seed)
    cam_sample = random.sample(cam_imgs, min(args.num_camera, len(cam_imgs)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Verwende {len(cam_sample)} Kamera-Bilder")
    print(f"[INFO] Output: {out_dir}")
    roi_cfg = config.get("camera", {}).get("card_roi", None)

    print(f"[INFO] Name-Crop: x={args.name_x}, y={args.name_y}, w={args.name_w}, h={args.name_h}")
    print(f"[INFO] Collector-Crop: x={args.collector_x}, y={args.collector_y}, w={args.collector_w}, h={args.collector_h}")
    print(f"[INFO] SetID-Crop: x={args.setid_x}, y={args.setid_y}, w={args.setid_w}, h={args.setid_h}")
    if args.show_roi and roi_cfg:
        print(f"[INFO] ROI aktiv: {roi_cfg}")

    process_set("camera", cam_sample, args, out_dir, roi_cfg if args.show_roi else None, args.show_roi)

    print("\n[FERTIG] Crops & Previews erzeugt. Passe Parameter an und erneut ausführen für Feintuning.")


if __name__ == "__main__":
    main()
