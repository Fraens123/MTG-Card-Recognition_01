from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.augmentations import CameraLikeAugmentor
from src.core.config_utils import load_config
from src.core.embedding_utils import build_card_embedding, compute_centroid, l2_normalize
from src.core.image_ops import crop_card_art, get_full_art_crop_cfg
from src.core.model_builder import load_encoder
from src.datasets.card_datasets import parse_scryfall_filename

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exportiert Embeddings fuer alle Scryfall-Karten.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Pfad zur Konfigurationsdatei")
    parser.add_argument(
        "--mode",
        choices=["runtime", "analysis"],
        default="runtime",
        help="runtime: 1 Zentroid pro Karte, analysis: alle Augs + optional Zentroid",
    )
    return parser.parse_args()


def _build_eval_transform(size_hw) -> T.Compose:
    return T.Compose(
        [
            T.Resize(size_hw, antialias=True),
            T.ToTensor(),
            T.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )


def _prepare_tensors(
    img: Image.Image,
    full_transform: T.Compose,
    full_crop_cfg: Dict,
    device: torch.device,
) -> torch.Tensor:
    full_img = crop_card_art(img, full_crop_cfg)
    full_tensor = full_transform(full_img).unsqueeze(0).to(device)
    return full_tensor


def _iterate_images(folder: str):
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            yield name, path


def _select_export_cfg(cfg: Dict, mode: str) -> tuple[Dict, str]:
    key = "embedding_export_analysis" if mode == "analysis" else "embedding_export_runtime"
    if key not in cfg:
        raise KeyError(f"Config-Block '{key}' fehlt.")
    return cfg[key] or {}, key


def _embed_card(
    model: torch.nn.Module,
    img: Image.Image,
    transform: T.Compose,
    crop_cfg: Dict,
    device: torch.device,
) -> np.ndarray:
    full_tensor = _prepare_tensors(img, transform, crop_cfg, device)
    embedding = build_card_embedding(model, full_tensor).cpu().numpy()
    return l2_normalize(embedding)


def _generate_augmented_images(
    base_img: Image.Image, num_augmentations: int, augmentor: CameraLikeAugmentor | None
) -> List[Image.Image]:
    if num_augmentations <= 0:
        return []
    if augmentor is None:
        return [base_img.copy() for _ in range(num_augmentations)]
    aug_list = augmentor.create_camera_like_augmentations(base_img, num_augmentations=num_augmentations)
    return aug_list[1 : num_augmentations + 1]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    export_cfg, export_key = _select_export_cfg(cfg, args.mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = export_cfg.get("model_path") or os.path.join(cfg.get("paths", {}).get("models_dir", ""), "encoder_fine.pt")
    print(f"[INFO] Exportiere Embeddings auf {device}")
    print(f"[LOAD] Modell: {model_path}")
    model = load_encoder(model_path, cfg, device=device)

    scryfall_dir = cfg["paths"]["scryfall_dir"]
    if not os.path.isdir(scryfall_dir):
        raise FileNotFoundError(f"scryfall_dir nicht gefunden: {scryfall_dir}")
    print(f"[DATA] Quelle: {scryfall_dir}")

    images_cfg = cfg.get("images", {})
    full_transform = _build_eval_transform(tuple(reversed(images_cfg.get("full_card_size", [224, 320]))))
    full_crop_cfg = get_full_art_crop_cfg(cfg)

    use_augmentations = bool(export_cfg.get("use_augmentations", False))
    num_augmentations = int(export_cfg.get("num_augmentations", 0))
    camera_aug_cfg = cfg.get("camera_augmentor", {})
    camera_augmentor: CameraLikeAugmentor | None = None
    if use_augmentations and num_augmentations > 0 and camera_aug_cfg.get("enabled", True):
        camera_augmentor = CameraLikeAugmentor(**camera_aug_cfg)

    cards: List[Dict[str, str]] = []
    embeddings: List[List[float]] = []
    files = list(_iterate_images(scryfall_dir))
    print(f"[INFO] Gefundene Kartenbilder: {len(files)}")
    for name, path in tqdm(files, desc="Exportiere Embeddings"):
        img = Image.open(path).convert("RGB")

        card_embeddings: List[np.ndarray] = []
        base_embedding = _embed_card(model, img, full_transform, full_crop_cfg, device)
        card_embeddings.append(base_embedding)

        if use_augmentations and num_augmentations > 0:
            aug_images = _generate_augmented_images(img, num_augmentations, camera_augmentor)
            for aug_img in aug_images:
                aug_embedding = _embed_card(model, aug_img, full_transform, full_crop_cfg, device)
                card_embeddings.append(aug_embedding)

        centroid = compute_centroid(card_embeddings)

        final_vectors: List[np.ndarray] = []
        if export_cfg.get("append_individual", False):
            final_vectors.extend(card_embeddings)
        if export_cfg.get("append_centroid", False):
            final_vectors.append(centroid)
        if export_cfg.get("use_centroid", False):
            final_vectors = [centroid]
        if not final_vectors:
            final_vectors = card_embeddings

        meta = parse_scryfall_filename(name)
        if meta:
            card_uuid, set_code, collector_number, card_name = meta
            card_name = card_name.replace("_", " ")
        else:
            card_uuid = os.path.splitext(name)[0]
            set_code = ""
            collector_number = ""
            card_name = card_uuid
        card_dict = {
            "card_uuid": card_uuid,
            "name": card_name,
            "set_code": set_code,
            "collector_number": collector_number,
            "image_path": str(path),
        }

        for vec in final_vectors:
            cards.append(card_dict)
            embeddings.append(vec.tolist())

    out_path = Path(export_cfg.get("output_path", os.path.join(cfg.get("paths", {}).get("embeddings_dir", "./embeddings"), "card_embeddings.json")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_cards = len(files)
    num_embeddings = len(embeddings)

    storage_desc = "unverändert"
    if export_cfg.get("use_centroid", False):
        storage_desc = "nur Zentroid"
    elif export_cfg.get("append_individual", False) and export_cfg.get("append_centroid", False):
        storage_desc = "Einzel + Zentroid"
    elif export_cfg.get("append_individual", False):
        storage_desc = "nur Einzel-Embeddings"
    elif export_cfg.get("append_centroid", False):
        storage_desc = "nur Zentroid angehängt"

    payload = {
        "cards": cards,
        "embeddings": embeddings,
        "meta": {
            "mode": args.mode,
            "config_section": export_key,
            "model_path": str(model_path),
            "scryfall_dir": scryfall_dir,
            "num_cards": num_cards,
            "num_embeddings": num_embeddings,
            "use_augmentations": use_augmentations,
            "num_augmentations": num_augmentations,
            "storage": storage_desc,
        },
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"[INFO] Embeddings exportiert nach {out_path}")
    print(f"[SUMMARY] Modus: {args.mode} | Karten/Cluster: {num_cards} | Embeddings geschrieben: {num_embeddings}")
    print(f"[SUMMARY] Augmentierungen: {'aktiv' if use_augmentations and num_augmentations > 0 else 'aus'} (n={num_augmentations})")
    print(f"[SUMMARY] Speicherlogik: {storage_desc}")
    print("[NEXT] Erkennung testen: python -m src.recognize_cards --config config.yaml")


if __name__ == "__main__":
    main()

