import argparse
import json
import os
import sys
from collections import OrderedDict
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
from src.core.embedding_utils import build_card_embedding
from src.core.image_ops import (
    crop_card_art,
    crop_name_field,
    crop_set_symbol,
    get_full_art_crop_cfg,
    get_name_field_crop_cfg,
    get_set_symbol_crop_cfg,
)
from src.core.model_builder import load_encoder
from src.datasets.card_datasets import parse_scryfall_filename

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exportiert Embeddings fuer alle Scryfall-Karten.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Pfad zur Konfigurationsdatei")
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
    symbol_transform: T.Compose,
    name_transform: T.Compose,
    full_crop_cfg: Dict,
    symbol_crop_cfg: Dict,
    name_crop_cfg: Dict,
    device: torch.device,
):
    full_tensor = full_transform(crop_card_art(img, full_crop_cfg)).unsqueeze(0).to(device)
    symbol_tensor = symbol_transform(crop_set_symbol(img, symbol_crop_cfg)).unsqueeze(0).to(device)
    name_tensor = name_transform(crop_name_field(img, name_crop_cfg)).unsqueeze(0).to(device)
    return full_tensor, symbol_tensor, name_tensor


def add_embedding(emb_list: List[List[float]], emb) -> None:
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    if isinstance(emb, np.ndarray):
        emb = emb.tolist()
    emb_list.append(emb)


def compute_centroid(emb_list: List[List[float]]):
    if not emb_list:
        return None
    arr = np.array(emb_list, dtype=np.float32)
    centroid = arr.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid.tolist()


def _iterate_images(folder: str):
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            yield name, path


def _generate_augmented_images(camera_aug, img: Image.Image, count: int):
    if camera_aug is None or count <= 0:
        return []
    aug_images = camera_aug.create_camera_like_augmentations(img, num_augmentations=count)
    return aug_images[1:]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    export_cfg = cfg.get("embedding_export", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_original_mode = bool(export_cfg.get("use_original_mode", False))
    use_cam_aug = bool(export_cfg.get("use_camera_augmentor", False))
    num_aug = max(0, int(export_cfg.get("num_augmentations", 0)))
    use_centroid = bool(export_cfg.get("use_centroid", False))
    append_centroid = bool(export_cfg.get("append_centroid", True))

    camera_aug = None
    if (not use_original_mode) and use_cam_aug and num_aug > 0:
        cam_cfg = dict(cfg.get("camera_augmentor", {}))
        cam_cfg.pop("enabled", None)
        camera_aug = CameraLikeAugmentor(**cam_cfg)

    if use_original_mode:
        mode_desc = "original"
    else:
        aug_label = "augmented" if camera_aug is not None else "no-aug"
        mode_desc = (
            f"{aug_label} (num_aug={num_aug if camera_aug is not None else 0}, "
            f"centroid={use_centroid}, append_centroid={append_centroid})"
        )
    print(f"[EXPORT] Embedding-Export-Modus: {mode_desc}")

    model_path = export_cfg.get("model_path") or os.path.join(cfg["paths"]["models_dir"], "encoder_fine.pt")
    print(f"[INFO] Exportiere Embeddings auf {device}")
    print(f"[LOAD] Modell: {model_path}")
    model = load_encoder(model_path, cfg, device=device)

    scryfall_dir = cfg["paths"]["scryfall_dir"]
    if not os.path.isdir(scryfall_dir):
        raise FileNotFoundError(f"scryfall_dir nicht gefunden: {scryfall_dir}")
    print(f"[DATA] Quelle: {scryfall_dir}")

    images_cfg = cfg.get("images", {})
    full_transform = _build_eval_transform(tuple(reversed(images_cfg.get("full_card_size", [224, 320]))))
    symbol_transform = _build_eval_transform(tuple(reversed(images_cfg.get("symbol_size", [160, 64]))))
    name_transform = _build_eval_transform(tuple(reversed(images_cfg.get("name_size", [64, 320]))))
    full_crop_cfg = get_full_art_crop_cfg(cfg)
    symbol_crop_cfg = get_set_symbol_crop_cfg(cfg)
    name_crop_cfg = get_name_field_crop_cfg(cfg)

    embedding_store: Dict[str, List] = OrderedDict()
    card_meta_store: Dict[str, Dict[str, str]] = OrderedDict()

    files = list(_iterate_images(scryfall_dir))
    print(f"[INFO] Gefundene Kartenbilder: {len(files)}")
    if use_original_mode:
        outputs_per_card = 1
    else:
        active_aug = num_aug if camera_aug is not None else 0
        base_aug_outputs = 1 + active_aug
        if use_centroid:
            outputs_per_card = 1 if not append_centroid else base_aug_outputs + 1
        else:
            outputs_per_card = base_aug_outputs
    expected_total = len(files) * outputs_per_card
    print(f"[INFO] Erwartete Embeddings (Schaetzung): {expected_total}")

    for name, path in tqdm(files, desc="Exportiere Embeddings"):
        img = Image.open(path).convert("RGB")
        full_tensor, symbol_tensor, name_tensor = _prepare_tensors(
            img,
            full_transform,
            symbol_transform,
            name_transform,
            full_crop_cfg,
            symbol_crop_cfg,
            name_crop_cfg,
            device,
        )
        base_embedding = build_card_embedding(model, full_tensor, symbol_tensor, name_tensor)

        meta = parse_scryfall_filename(name)
        if meta:
            card_uuid, set_code, collector_number, card_name = meta
            card_name = card_name.replace("_", " ")
        else:
            card_uuid = os.path.splitext(name)[0]
            set_code = ""
            collector_number = ""
            card_name = card_uuid

        card_meta_store.setdefault(
            card_uuid,
            {
                "card_uuid": card_uuid,
                "name": card_name,
                "set_code": set_code,
                "collector_number": collector_number,
                "image_path": str(path),
            },
        )

        emb_list: List[List[float]] = []
        add_embedding(emb_list, base_embedding)

        if (not use_original_mode) and (camera_aug is not None):
            for aug_img in _generate_augmented_images(camera_aug, img, num_aug):
                full_tensor, symbol_tensor, name_tensor = _prepare_tensors(
                    aug_img,
                    full_transform,
                    symbol_transform,
                    name_transform,
                    full_crop_cfg,
                    symbol_crop_cfg,
                    name_crop_cfg,
                    device,
                )
                aug_embedding = build_card_embedding(model, full_tensor, symbol_tensor, name_tensor)
                add_embedding(emb_list, aug_embedding)

        centroid = None
        if (not use_original_mode) and use_centroid:
            centroid = compute_centroid(emb_list)
            if centroid is None and emb_list:
                centroid = emb_list[0]

        if use_original_mode:
            embedding_store[card_uuid] = emb_list[0]
        else:
            if use_centroid and not append_centroid:
                embedding_store[card_uuid] = centroid
            else:
                if use_centroid and append_centroid and centroid is not None:
                    add_embedding(emb_list, centroid)
                embedding_store[card_uuid] = emb_list

    cards: List[Dict[str, str]] = []
    embeddings: List[List[float]] = []
    for card_uuid, meta in card_meta_store.items():
        stored = embedding_store.get(card_uuid)
        if stored is None:
            continue
        if isinstance(stored, list) and stored and isinstance(stored[0], (float, int)):
            embed_iter = [stored]
        elif isinstance(stored, list):
            embed_iter = stored
        else:
            embed_iter = [stored]
        for emb in embed_iter:
            if emb is None:
                continue
            cards.append(dict(meta))
            embeddings.append(emb)

    out_dir = cfg["paths"]["embeddings_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "card_embeddings.json")
    payload = {
        "cards": cards,
        "embeddings": embeddings,
        "meta": {"model_path": model_path, "scryfall_dir": scryfall_dir},
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"[INFO] Tatsaechliche Embeddings: {len(embeddings)}")
    print(f"[INFO] Embeddings exportiert nach {out_path}")
    print("[NEXT] Erkennung testen: python -m src.recognize_cards --config config.yaml")


if __name__ == "__main__":
    main()

