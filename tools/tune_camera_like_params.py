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
import optuna

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
        default=80,
        help="Anzahl Bayesian Optimization Trials (Standard: 80, empfohlen 50-100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optionaler Zufalls-Seed fuer reproduzierbare Optimierung.",
    )
    parser.add_argument(
        "--use-random-search",
        action="store_true",
        help="Fallback auf alten Random-Search statt Bayesian Optimization.",
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


def sample_params_with_optuna(trial: optuna.Trial, base_aug: Dict) -> Dict:
    """
    Bayesian Optimization: Optuna schlägt Parameter vor basierend auf bisherigen Ergebnissen.
    Fokussiert automatisch auf vielversprechende Bereiche.
    
    Ranges optimiert für Pi-Cam: enge Bereiche um realistische Werte.
    """
    # Sehr enge Ranges für beste Ergebnisse (basierend auf typischen Pi-Cam-Eigenschaften)
    sampled = {
        "brightness": (
            trial.suggest_float("brightness_min", 0.75, 0.90, step=0.01),
            trial.suggest_float("brightness_max", 1.05, 1.25, step=0.01),
        ),
        "contrast": (
            trial.suggest_float("contrast_min", 0.85, 0.95, step=0.01),
            trial.suggest_float("contrast_max", 1.0, 1.15, step=0.01),
        ),
        "saturation": (
            trial.suggest_float("saturation_min", 0.70, 0.85, step=0.01),
            trial.suggest_float("saturation_max", 1.10, 1.40, step=0.01),
        ),
        "color_temperature_range": (
            trial.suggest_float("color_temp_min", 0.60, 0.80, step=0.01),
            trial.suggest_float("color_temp_max", 1.05, 1.25, step=0.01),
        ),
        "gamma_range": (
            trial.suggest_float("gamma_min", 0.85, 0.95, step=0.01),
            trial.suggest_float("gamma_max", 0.95, 1.10, step=0.01),
        ),
        "rotation_deg": trial.suggest_float("rotation_deg", 0.0, 4.0, step=0.1),
        "perspective": trial.suggest_float("perspective", 0.0, 0.12, step=0.005),
        "shadow": trial.suggest_float("shadow", 0.05, 0.40, step=0.01),
        "noise_std": trial.suggest_float("noise_std", 0.005, 0.025, step=0.001),
        "blur_prob": trial.suggest_float("blur_prob", 0.05, 0.25, step=0.01),
    }
    
    # Validierung: min < max
    for key in ["brightness", "contrast", "saturation", "color_temperature_range", "gamma_range"]:
        min_val, max_val = sampled[key]
        if min_val >= max_val:
            sampled[key] = (min_val, min_val + 0.05)
    
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
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("hardware", {}).get("use_cuda", True) else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Optimization: {'Random Search (Legacy)' if args.use_random_search else 'Bayesian (Optuna)'}")

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

    # Closure für Optuna objective function
    def objective(trial: optuna.Trial) -> float:
        sampled = sample_params_with_optuna(trial, base_aug_cfg)
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
        
        # Log Zwischenstand
        if trial.number % 10 == 0 or trial.number < 5:
            print(f"[Trial {trial.number:03d}] score={mean_cos:.4f}")
        
        return mean_cos  # Optuna maximiert

    # Bayesian Optimization
    print(f"[RUN] Starte Bayesian Optimization mit {args.samples} Trials ...")
    print("[INFO] Optuna findet automatisch vielversprechende Bereiche!\n")
    
    sampler = optuna.samplers.TPESampler(seed=args.seed) if args.seed else optuna.samplers.TPESampler()
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="camera_param_tuning"
    )
    
    study.optimize(objective, n_trials=args.samples, show_progress_bar=True)

    # Beste Ergebnisse
    print(f"\n[RESULT] Beste Score: {study.best_value:.4f}")
    print(f"[RESULT] Gefunden in Trial #{study.best_trial.number}")
    
    # Rekonstruiere beste Config
    best_params = study.best_params
    best_cfg = copy.deepcopy(base_aug_cfg)
    best_cfg.update({
        "brightness": (best_params["brightness_min"], best_params["brightness_max"]),
        "contrast": (best_params["contrast_min"], best_params["contrast_max"]),
        "saturation": (best_params["saturation_min"], best_params["saturation_max"]),
        "color_temperature_range": (best_params["color_temp_min"], best_params["color_temp_max"]),
        "gamma_range": (best_params["gamma_min"], best_params["gamma_max"]),
        "rotation_deg": best_params["rotation_deg"],
        "perspective": best_params["perspective"],
        "shadow": best_params["shadow"],
        "noise_std": best_params["noise_std"],
        "blur_prob": best_params["blur_prob"],
    })

    # Top 10 Trials
    print("\n[TOP 10] Beste Parameter-Sets:")
    for i, trial in enumerate(sorted(study.trials, key=lambda t: t.value if t.value else -1, reverse=True)[:10], start=1):
        if trial.value is None:
            continue
        print(f"  #{i:02d} | score={trial.value:.4f} | Trial {trial.number}")

    print("\n[FINAL] Bestes Parameter-Set (Augment-Block für YAML):\n")
    yaml_block = format_yaml_block(best_cfg)
    print(yaml_block)
    
    print("\n[TIP] Kopiere den YAML-Block in deine config.train500.yaml unter training.fine.augment")
    print(f"[TIP] Mit {args.samples} Trials sollte Score >{study.best_value:.2f} erreichbar sein.")


if __name__ == "__main__":
    main()
