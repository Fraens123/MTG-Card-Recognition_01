import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_utils import load_config
from src.core.embedding_utils import build_card_embedding
from src.core.image_ops import crop_card_art, crop_set_symbol, get_full_art_crop_cfg, get_set_symbol_crop_cfg
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


def _iterate_images(folder: str):
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            yield name, path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(cfg["paths"]["models_dir"], "encoder_fine.pt")
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
    full_crop_cfg = get_full_art_crop_cfg(cfg)
    symbol_crop_cfg = get_set_symbol_crop_cfg(cfg)

    cards: List[Dict[str, str]] = []
    embeddings: List[List[float]] = []
    files = list(_iterate_images(scryfall_dir))
    print(f"[INFO] Gefundene Kartenbilder: {len(files)}")
    for name, path in tqdm(files, desc="Exportiere Embeddings"):
        img = Image.open(path).convert("RGB")
        full_tensor = full_transform(crop_card_art(img, full_crop_cfg)).unsqueeze(0).to(device)
        symbol_tensor = symbol_transform(crop_set_symbol(img, symbol_crop_cfg)).unsqueeze(0).to(device)
        embedding = build_card_embedding(model, full_tensor, symbol_tensor).cpu().numpy().tolist()
        meta = parse_scryfall_filename(name)
        if meta:
            card_uuid, set_code, collector_number, card_name = meta
            card_name = card_name.replace("_", " ")
        else:
            card_uuid = os.path.splitext(name)[0]
            set_code = ""
            collector_number = ""
            card_name = card_uuid
        cards.append(
            {
                "card_uuid": card_uuid,
                "name": card_name,
                "set_code": set_code,
                "collector_number": collector_number,
                "image_path": str(path),
            }
        )
        embeddings.append(embedding)

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
    print(f"[INFO] Embeddings exportiert nach {out_path}")
    print("[NEXT] Erkennung testen: python -m src.recognize_cards --config config.yaml")


if __name__ == "__main__":
    main()


