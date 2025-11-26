from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

# Resolve repository root (../ from tools/)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.augmentations import CameraLikeAugmentor
from src.core.config_utils import load_config
from src.core.embedding_utils import build_card_embedding, l2_normalize
from src.core.image_ops import (
    build_resize_normalize_transform,
    crop_card_art,
    get_full_art_crop_cfg,
    resolve_resize_hw,
)
from src.core.model_builder import load_encoder
from src.datasets.card_datasets import _map_camera_aug_params as map_camera_aug_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune CameraLikeAugmentor parameters against Pi-Cam pairs.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.train20k.yaml",
        help="Pfad zur YAML-Config (z.B. config.train20k.yaml).",
    )
    parser.add_argument(
        "--pairs-file",
        type=str,
        required=True,
        help="JSON mit Bildpaaren (Felder: 'pi', 'scry'). Pfade relativ zum Repo-Root moeglich.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Anzahl Random-Samples fuer die Parameter-Suche.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optionaler Zufalls-Seed fuer reproduzierbare Parameter-Samples.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def load_pairs(pairs_path: Path) -> List[Dict[str, Path]]:
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs-File nicht gefunden: {pairs_path}")
    with pairs_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "pairs" in data:
        entries = data.get("pairs", [])
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("Pairs-File muss eine Liste oder ein Dict mit Key 'pairs' enthalten.")

    pairs: List[Dict[str, Path]] = []
    for entry in entries:
        if not isinstance(entry, dict) or "pi" not in entry or "scry" not in entry:
            continue
        pi_path = resolve_path(str(entry["pi"]))
        scry_path = resolve_path(str(entry["scry"]))
        if not pi_path.exists():
            print(f"[WARN] Pi-Bild fehlt, ueberspringe: {pi_path}")
            continue
        if not scry_path.exists():
            print(f"[WARN] Scryfall-Bild fehlt, ueberspringe: {scry_path}")
            continue
        pairs.append({"pi": pi_path, "scry": scry_path})
    if not pairs:
        raise RuntimeError("Keine gueltigen Paare gefunden.")
    return pairs


def crop_card_roi(img: Image.Image, roi_cfg: Optional[Dict]) -> Image.Image:
    """Karten-ROI wie im Runtime-Pfad schneiden."""
    if not roi_cfg:
        return img
    w, h = img.size
    try:
        x_min = float(roi_cfg.get("x_min", 0.0))
        y_min = float(roi_cfg.get("y_min", 0.0))
        x_max = float(roi_cfg.get("x_max", 1.0))
        y_max = float(roi_cfg.get("y_max", 1.0))
    except Exception:
        return img
    x0 = int(x_min * w)
    y0 = int(y_min * h)
    x1 = int(x_max * w)
    y1 = int(y_max * h)
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return img.crop((x0, y0, x1, y1))


def apply_table_background(card_img: Image.Image, table_img=None, scale: float = 1.0, offset_x: int = 0, offset_y: int = 0) -> Image.Image:
    """
    Platzhalter: hier koennte spaeter ein Tisch-Hintergrund eingefuegt werden.
    Aktuell wird das Kartenbild unveraendert zurueckgegeben.
    """
    return card_img


def resolve_model_path(cfg: Dict) -> str:
    paths_cfg = cfg.get("paths", {})
    runtime_cfg = cfg.get("embedding_export_runtime", {})
    if runtime_cfg.get("model_path"):
        return runtime_cfg["model_path"]
    fine_cfg = cfg.get("training", {}).get("fine", {})
    model_filename = fine_cfg.get("model_filename", "encoder_fine.pt")
    return os.path.join(paths_cfg.get("models_dir", "./models"), model_filename)


def compute_embedding(
    model: torch.nn.Module,
    img: Image.Image,
    transform,
    device: torch.device,
) -> np.ndarray:
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = build_card_embedding(model, tensor).cpu().numpy().flatten()
    return l2_normalize(emb)


def compute_pi_embeddings(
    pairs: Sequence[Dict[str, Path]],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    art_crop_cfg: Optional[Dict],
    card_roi_cfg: Optional[Dict],
) -> Dict[Path, np.ndarray]:
    pi_embeddings: Dict[Path, np.ndarray] = {}
    for entry in pairs:
        pi_path = entry["pi"]
        if pi_path in pi_embeddings:
            continue
        with Image.open(pi_path) as img_raw:
            img = img_raw.convert("RGB")
        img = crop_card_roi(img, card_roi_cfg)
        img = crop_card_art(img, art_crop_cfg)
        pi_embeddings[pi_path] = compute_embedding(model, img, transform, device)
    return pi_embeddings


def load_scryfall_images(pairs: Sequence[Dict[str, Path]]) -> Dict[Path, Image.Image]:
    cache: Dict[Path, Image.Image] = {}
    for entry in pairs:
        scry_path = entry["scry"]
        if scry_path in cache:
            continue
        with Image.open(scry_path) as img_raw:
            cache[scry_path] = img_raw.convert("RGB")
    return cache


def _jitter_range(base: Tuple[float, float], rel: float, clamp: Tuple[float, float], rng: random.Random) -> Tuple[float, float]:
    lo, hi = base
    lo_new = lo * rng.uniform(1.0 - rel, 1.0 + rel)
    hi_new = hi * rng.uniform(1.0 - rel, 1.0 + rel)
    lo_new, hi_new = sorted((lo_new, hi_new))
    lo_new = max(clamp[0], lo_new)
    hi_new = min(clamp[1], hi_new)
    if hi_new <= lo_new:
        hi_new = lo_new + 1e-4
    return (lo_new, hi_new)


def _jitter_scalar(base: float, rel: float, clamp: Tuple[float, float], rng: random.Random) -> float:
    sampled = base * rng.uniform(1.0 - rel, 1.0 + rel)
    return float(min(clamp[1], max(clamp[0], sampled)))


def sample_tunable_params(base_aug: Dict, rng: random.Random) -> Dict:
    """Sample um die Basis-Konfiguration herum (moderate Jitter)."""
    brightness = tuple(base_aug.get("brightness", (0.85, 1.2)))
    contrast = tuple(base_aug.get("contrast", (0.85, 1.2)))
    saturation = tuple(base_aug.get("saturation", (0.7, 1.2)))
    color_temp = tuple(base_aug.get("color_temperature_range", (0.75, 1.25)))
    gamma = tuple(base_aug.get("gamma_range", (0.85, 1.15)))

    rotation = float(base_aug.get("rotation_deg", 4.0))
    perspective = float(base_aug.get("perspective", 0.10))
    shadow = float(base_aug.get("shadow", 0.30))
    noise_std = float(base_aug.get("noise_std", 0.02))
    blur_prob = float(base_aug.get("blur_prob", 0.25))

    sampled = {
        "brightness": _jitter_range(brightness, rel=0.15, clamp=(0.3, 1.8), rng=rng),
        "contrast": _jitter_range(contrast, rel=0.15, clamp=(0.3, 1.8), rng=rng),
        "saturation": _jitter_range(saturation, rel=0.2, clamp=(0.2, 1.8), rng=rng),
        "color_temperature_range": _jitter_range(color_temp, rel=0.2, clamp=(0.3, 2.5), rng=rng),
        "gamma_range": _jitter_range(gamma, rel=0.2, clamp=(0.5, 2.0), rng=rng),
        "rotation_deg": _jitter_scalar(rotation, rel=0.25, clamp=(0.0, 12.0), rng=rng),
        "perspective": _jitter_scalar(perspective, rel=0.35, clamp=(0.0, 0.25), rng=rng),
        "shadow": _jitter_scalar(shadow, rel=0.35, clamp=(0.0, 0.9), rng=rng),
        "noise_std": _jitter_scalar(noise_std, rel=0.5, clamp=(0.0, 0.12), rng=rng),
        "blur_prob": _jitter_scalar(blur_prob, rel=0.5, clamp=(0.0, 0.95), rng=rng),
    }
    return sampled


def build_augmentor(candidate_aug_cfg: Dict) -> CameraLikeAugmentor:
    params = map_camera_aug_params(candidate_aug_cfg)
    # Uebernehme explizit gesetzte Wahrscheinlichkeiten aus der Config (falls vorhanden)
    for prob_key in (
        "brightness_prob",
        "contrast_prob",
        "saturation_prob",
        "gamma_prob",
        "color_temperature_prob",
        "hue_shift_prob",
        "white_balance_prob",
        "sharpness_prob",
        "vignette_prob",
        "chromatic_aberration_prob",
        "blur_prob",
        "noise_prob",
        "rotation_prob",
        "perspective_prob",
        "shadow_prob",
    ):
        if prob_key in candidate_aug_cfg:
            params[prob_key] = candidate_aug_cfg[prob_key]
    return CameraLikeAugmentor(**params)


def evaluate_params(
    augmentor: CameraLikeAugmentor,
    pairs: Sequence[Dict[str, Path]],
    scry_images: Dict[Path, Image.Image],
    pi_embeddings: Dict[Path, np.ndarray],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    art_crop_cfg: Optional[Dict],
) -> float:
    sims: List[float] = []
    with torch.no_grad():
        for entry in pairs:
            pi_path = entry["pi"]
            base_img = scry_images[entry["scry"]]
            aug_img = augmentor(base_img.copy())
            aug_img = apply_table_background(aug_img, table_img=None, scale=1.0, offset_x=0, offset_y=0)
            aug_img = crop_card_art(aug_img, art_crop_cfg)
            scry_emb = compute_embedding(model, aug_img, transform, device)
            sims.append(float(np.dot(scry_emb, pi_embeddings[pi_path])))
    return float(np.mean(sims)) if sims else 0.0


def format_yaml_block(best_aug: Dict) -> str:
    ordered_keys = [
        "camera_like",
        "camera_like_strength",
        "brightness",
        "contrast",
        "saturation",
        "color_temperature_range",
        "gamma_range",
        "rotation_deg",
        "perspective",
        "shadow",
        "noise_std",
        "blur_prob",
    ]
    lines = ["training:", "  fine:", "    augment:"]
    for key in ordered_keys:
        if key not in best_aug:
            continue
        val = best_aug[key]
        if isinstance(val, (list, tuple)):
            val_str = f"[{val[0]:.4f}, {val[1]:.4f}]"
        elif isinstance(val, float):
            val_str = f"{val:.4f}"
        else:
            val_str = str(val).lower() if isinstance(val, bool) else str(val)
        lines.append(f"      {key}: {val_str}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("hardware", {}).get("use_cuda", True) else "cpu")
    print(f"[INFO] Device: {device}")

    model_path = resolve_model_path(cfg)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Encoder-Gewicht nicht gefunden: {model_path}")
    print(f"[LOAD] Modell: {model_path}")
    model = load_encoder(model_path, cfg, device=device)
    model.eval()

    resize_hw = resolve_resize_hw(cfg, cfg.get("paths", {}).get("scryfall_dir"))
    transform = build_resize_normalize_transform(resize_hw)
    art_crop_cfg = get_full_art_crop_cfg(cfg)
    card_roi_cfg = cfg.get("camera", {}).get("card_roi")
    print(f"[INFO] Resize-Ziel (H,W): {resize_hw}")

    pairs = load_pairs(resolve_path(args.pairs_file))
    print(f"[INFO] Lade {len(pairs)} Paare aus {args.pairs_file}")

    print("[INFO] Berechne Pi-Cam-Embeddings einmalig ...")
    t0 = time.time()
    pi_embeddings = compute_pi_embeddings(pairs, model, transform, device, art_crop_cfg, card_roi_cfg)
    print(f"[INFO] Fertig ({len(pi_embeddings)} Embeddings) in {(time.time() - t0):.2f}s")

    print("[INFO] Lade Scryfall-Bilder in den RAM ...")
    scry_images = load_scryfall_images(pairs)
    print(f"[INFO] Geladen: {len(scry_images)} Scryfall-Bilder")

    base_aug_cfg = copy.deepcopy(cfg.get("training", {}).get("fine", {}).get("augment", {}))
    base_aug_cfg.setdefault("camera_like", True)
    base_aug_cfg.setdefault("camera_like_strength", 1.0)

    results: List[Dict] = []
    best_mean = -1.0
    best_cfg: Optional[Dict] = None

    print(f"[RUN] Starte Random-Search mit {args.samples} Samples ...")
    for idx in range(1, args.samples + 1):
        sampled = sample_tunable_params(base_aug_cfg, rng)
        candidate_aug_cfg = copy.deepcopy(base_aug_cfg)
        candidate_aug_cfg.update(sampled)

        augmentor = build_augmentor(candidate_aug_cfg)
        mean_cos = evaluate_params(
            augmentor,
            pairs,
            scry_images,
            pi_embeddings,
            model,
            transform,
            device,
            art_crop_cfg,
        )
        results.append({"mean_cos": mean_cos, "cfg": candidate_aug_cfg})
        if mean_cos > best_mean:
            best_mean = mean_cos
            best_cfg = candidate_aug_cfg
        print(f"[{idx:03d}/{args.samples}] mean_cos={mean_cos:.4f} | best={best_mean:.4f}")

    results.sort(key=lambda r: r["mean_cos"], reverse=True)
    top_n = results[: min(10, len(results))]

    print("\n[RESULT] Top Parameter-Sets:")
    for i, entry in enumerate(top_n, start=1):
        cfg_short = entry["cfg"]
        print(
            f"  #{i:02d} | mean_cos={entry['mean_cos']:.4f} | "
            f"brightness={cfg_short['brightness']} | contrast={cfg_short['contrast']} | "
            f"saturation={cfg_short['saturation']} | gamma={cfg_short['gamma_range']} | "
            f"temp={cfg_short['color_temperature_range']} | rot={cfg_short['rotation_deg']:.2f} | "
            f"persp={cfg_short['perspective']:.3f} | shadow={cfg_short['shadow']:.3f} | "
            f"noise_std={cfg_short['noise_std']:.4f} | blur_prob={cfg_short['blur_prob']:.3f}"
        )

    if best_cfg is None:
        print("[ERROR] Keine gueltigen Ergebnisse.")
        return

    print("\n[FINAL] Bestes Parameter-Set (Augment-Block fuer YAML):\n")
    yaml_block = format_yaml_block(best_cfg)
    print(yaml_block)


if __name__ == "__main__":
    main()
