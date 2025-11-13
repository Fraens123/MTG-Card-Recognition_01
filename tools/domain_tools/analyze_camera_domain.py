from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import sys
import numpy as np
from PIL import Image
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()]


def sample_images(images: List[Path], limit: int) -> List[Path]:
    if limit <= 0 or len(images) <= limit:
        return images
    rng = random.Random(42)
    return rng.sample(images, limit)


def laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    core = gray[1:-1, 1:-1]
    lap = (
        -4 * core
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def compute_stats(image_paths: List[Path]) -> Dict[str, float]:
    if not image_paths:
        return {}

    luminance_means: List[float] = []
    luminance_stds: List[float] = []
    channel_means = []
    lap_vars: List[float] = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        arr = np.asarray(img).astype(np.float32) / 255.0
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b

        luminance_means.append(float(luminance.mean()))
        luminance_stds.append(float(luminance.std()))
        channel_means.append(
            (
                float(r.mean()),
                float(g.mean()),
                float(b.mean()),
            )
        )
        lap_vars.append(laplacian_variance(luminance))

    mean_r = float(np.mean([c[0] for c in channel_means]))
    mean_g = float(np.mean([c[1] for c in channel_means]))
    mean_b = float(np.mean([c[2] for c in channel_means]))

    return {
        "count": len(channel_means),
        "mean_luminance": float(np.mean(luminance_means)),
        "std_luminance": float(np.mean(luminance_stds)),
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "laplacian_var": float(np.mean(lap_vars)),
    }


def print_stats(label: str, stats: Dict[str, float]) -> None:
    if not stats:
        print(f"[WARN] No data for {label}")
        return
    print(f"\n=== {label} ===")
    print(f"Samples:           {stats['count']}")
    print(f"Mean luminance:    {stats['mean_luminance']:.4f}")
    print(f"Luminance std:     {stats['std_luminance']:.4f}")
    print(f"Mean RGB:          ({stats['mean_r']:.4f}, {stats['mean_g']:.4f}, {stats['mean_b']:.4f})")
    print(f"Laplacian variance:{stats['laplacian_var']:.4f}")


def suggest_ranges(scryfall: Dict[str, float], camera: Dict[str, float]) -> None:
    if not scryfall or not camera:
        return
    print("\n=== Suggested CameraLikeAugmentor ranges ===")
    brightness_ratio = camera["mean_luminance"] / max(scryfall["mean_luminance"], 1e-6)
    brightness_min = max(0.4, brightness_ratio - camera["std_luminance"])
    brightness_max = min(1.6, brightness_ratio + camera["std_luminance"])
    print(f"brightness_min/max ~= ({brightness_min:.2f}, {brightness_max:.2f})")

    contrast_ratio = camera["std_luminance"] / max(scryfall["std_luminance"], 1e-6)
    contrast_min = max(0.6, contrast_ratio * 0.8)
    contrast_max = min(1.6, contrast_ratio * 1.2)
    print(f"contrast_min/max   ~= ({contrast_min:.2f}, {contrast_max:.2f})")

    blur_hint = math.sqrt(camera["laplacian_var"] + 1e-6)
    blur_sigma = max(0.2, 1.0 / (blur_hint + 1e-6))
    print(f"blur_sigma_max     ~= {blur_sigma:.2f}")

    noise_hint = max(0.005, abs(camera["mean_luminance"] - scryfall["mean_luminance"]) / 3)
    print(f"noise_std_max      ~= {noise_hint:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse Scryfall vs. Kamera-Domaene.")
    parser.add_argument("--config", default="config.yaml", help="Pfad zur config.yaml")
    parser.add_argument("--scryfall-limit", type=int, default=200, help="Max. Scryfall-Bilder fuer Statistik")
    parser.add_argument("--camera-limit", type=int, default=200, help="Max. Kamera-Bilder fuer Statistik")
    args = parser.parse_args()

    config = load_config(args.config)
    scryfall_dir = Path(config["data"]["scryfall_images"])
    camera_dir = Path(config["data"]["camera_images"])

    scryfall_images = sample_images(list_images(scryfall_dir), args.scryfall_limit)
    camera_images = sample_images(list_images(camera_dir), args.camera_limit)

    scry_stats = compute_stats(scryfall_images)
    cam_stats = compute_stats(camera_images)

    print_stats("Scryfall", scry_stats)
    print_stats("Pi-Cam", cam_stats)
    suggest_ranges(scry_stats, cam_stats)


if __name__ == "__main__":
    main()
