from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from src.core.embedding_utils import build_card_embedding_batch, compute_centroid, l2_normalize
from src.core.image_ops import crop_card_art, get_full_art_crop_cfg
from src.core.model_builder import load_encoder
from src.core.sqlite_store import SqliteEmbeddingStore
from src.datasets.card_datasets import parse_scryfall_filename

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exportiert Embeddings fuer alle Scryfall-Karten.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.train20k.yaml",
        help="Pfad zur Konfigurationsdatei (Default: config.train20k.yaml)",
    )
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


def _generate_augmented_images(
    base_img: Image.Image, num_augmentations: int, augmentor: CameraLikeAugmentor | None
) -> List[Image.Image]:
    if num_augmentations <= 0:
        return []
    if augmentor is None:
        return [base_img.copy() for _ in range(num_augmentations)]
    aug_list = augmentor.create_camera_like_augmentations(base_img, num_augmentations=num_augmentations)
    return aug_list[1 : num_augmentations + 1]


def _prepare_card_tensors(
    name: str,
    path: str,
    transform: T.Compose,
    crop_cfg: Dict,
    use_augs: bool,
    num_aug: int,
    camera_aug_cfg: Dict,
) -> Tuple[Dict[str, str], List[torch.Tensor]]:
    """Bereitet Tensoren für eine Karte vor.
    
    Returns:
        card_dict mit scryfall_id (als card_uuid), Liste von Tensoren
    """
    img = Image.open(path).convert("RGB")
    card_embeddings_tensors: List[torch.Tensor] = []

    base_tensor = _prepare_tensors(img, transform, crop_cfg, device=torch.device("cpu")).squeeze(0)
    card_embeddings_tensors.append(base_tensor)

    if use_augs and num_aug > 0:
        augmentor = CameraLikeAugmentor(**camera_aug_cfg) if camera_aug_cfg.get("enabled", True) else None
        for aug_img in _generate_augmented_images(img, num_aug, augmentor):
            aug_tensor = _prepare_tensors(aug_img, transform, crop_cfg, device=torch.device("cpu")).squeeze(0)
            card_embeddings_tensors.append(aug_tensor)

    # Parse Dateinamen um Scryfall-ID zu extrahieren
    meta = parse_scryfall_filename(name)
    if meta:
        scryfall_id, set_code, collector_number, card_name = meta
        card_name = card_name.replace("_", " ")
    else:
        # Fallback wenn Dateiname nicht parsbar
        scryfall_id = os.path.splitext(name)[0]
        set_code = ""
        collector_number = ""
        card_name = scryfall_id
    
    card_dict = {
        "card_uuid": scryfall_id,  # card_uuid = scryfall_id
        "name": card_name,
        "set_code": set_code,
        "collector_number": collector_number,
        "image_path": str(path),
    }
    return card_dict, card_embeddings_tensors


def _process_batch(
    pending: List[Tuple[torch.Tensor, str, str]],
    model: torch.nn.Module,
    device: torch.device,
) -> List[Tuple[np.ndarray, str, str]]:
    if not pending:
        return []
    batch_tensors, batch_ids, batch_paths = zip(*pending)
    full_batch = torch.stack(batch_tensors)
    if device.type == "cuda":
        full_batch = full_batch.pin_memory()
    full_batch = full_batch.to(device, non_blocking=True)
    emb_batch = build_card_embedding_batch(model, full_batch).cpu().numpy()
    return [(l2_normalize(vec), cid, path) for vec, cid, path in zip(emb_batch, batch_ids, batch_paths)]


def _expected_vectors_per_card(cfg: Dict, base_count: int) -> int:
    expected = 0
    if cfg.get("append_individual", False):
        expected += base_count
    if cfg.get("append_centroid", False):
        expected += 1
    if cfg.get("use_centroid", False):
        expected = 1
    if expected == 0:
        expected = base_count
    return expected


def _ensure_relative_path(path: str) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return Path(path).as_posix()


def _infer_language_from_filename(filename: str) -> Optional[str]:
    parts = Path(filename).stem.split("_")
    if len(parts) >= 4:
        lang = parts[-2]
        if len(lang) == 2:
            return lang.lower()
    return None


def _get_oracle_id_from_db(store: SqliteEmbeddingStore, scryfall_id: str) -> Optional[str]:
    """Holt die oracle_id aus der karten-Tabelle für eine gegebene Scryfall-ID.
    
    Args:
        store: SqliteEmbeddingStore Instanz
        scryfall_id: Die Scryfall-ID (Print-ID)
        
    Returns:
        oracle_id oder None falls nicht gefunden
    """
    import sqlite3
    try:
        with sqlite3.connect(store.db_path) as conn:
            cur = conn.execute("SELECT oracle_id FROM karten WHERE id = ?", (scryfall_id,))
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
    except Exception:
        pass
    return None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    export_cfg, export_key = _select_export_cfg(cfg, args.mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = export_cfg.get("model_path") or os.path.join(cfg.get("paths", {}).get("models_dir", ""), "encoder_fine.pt")
    print(f"[INFO] Exportiere Embeddings auf {device}")
    print(f"[LOAD] Modell: {model_path}")
    model = load_encoder(model_path, cfg, device=device)
    model.eval()

    scryfall_dir = cfg["paths"]["scryfall_dir"]
    if not os.path.isdir(scryfall_dir):
        raise FileNotFoundError(f"scryfall_dir nicht gefunden: {scryfall_dir}")
    print(f"[DATA] Quelle: {scryfall_dir}")

    images_cfg = cfg.get("images", {})
    full_transform = _build_eval_transform(tuple(reversed(images_cfg.get("full_card_size", [224, 320]))))
    full_crop_cfg = get_full_art_crop_cfg(cfg)

    use_augmentations = bool(export_cfg.get("use_augmentations", False))
    num_augmentations = int(export_cfg.get("num_augmentations", 0))
    export_batch_size = int(export_cfg.get("export_batch_size", 64))
    num_workers = int(export_cfg.get("num_workers", 4))
    camera_aug_cfg = cfg.get("camera_augmentor", {})
    use_centroid = bool(export_cfg.get("use_centroid", False))
    append_individual = bool(export_cfg.get("append_individual", False))
    append_centroid = bool(export_cfg.get("append_centroid", False))
    if not use_centroid and not append_individual and not append_centroid:
        append_individual = True  # Fallback: wie bisher nur Einzel-Embeddings

    files = list(_iterate_images(scryfall_dir))
    print(f"[INFO] Gefundene Kartenbilder: {len(files)}")
    card_meta: Dict[str, Dict[str, str]] = {}
    pending: List[Tuple[torch.Tensor, str, str]] = []
    centroid_sum: Dict[str, np.ndarray] = {}
    centroid_count: Dict[str, int] = {}
    # Streaming: wir schreiben Einzel-Embeddings sofort und akkumulieren nur Summen/Counts für Zentroiden
    sqlite_path = cfg.get("database", {}).get("sqlite_path", "tcg_database/database/karten.db")
    emb_dim = int(cfg.get("encoder", {}).get("emb_dim", cfg.get("model", {}).get("embed_dim", 1024)))
    store = SqliteEmbeddingStore(sqlite_path, emb_dim=emb_dim)
    store.clear_embeddings(args.mode)
    image_cache: Dict[Tuple[str, str], int] = {}
    oracle_cache: Dict[str, str] = {}
    aug_index_counter: Dict[str, int] = {}
    num_embeddings = 0
    print(f"[STORE] Schreibe Embeddings nach {sqlite_path} (mode={args.mode})")

    def _flush_pending():
        nonlocal pending, num_embeddings
        if not pending:
            return
        batch_results = _process_batch(pending, model, device)
        pending = []
        for vec_np, cid, img_path in batch_results:
            # Zentroid-Akkumulation nur, wenn wir sie brauchen
            if use_centroid or append_centroid:
                if cid not in centroid_sum:
                    centroid_sum[cid] = vec_np.astype(np.float32, copy=True)
                    centroid_count[cid] = 1
                else:
                    centroid_sum[cid] += vec_np
                    centroid_count[cid] += 1

            # Einzel-Embeddings sofort speichern, sofern gewünscht und nicht use_centroid-only
            need_individuals = append_individual and not use_centroid
            if need_individuals:
                oracle_id = oracle_cache.get(cid)
                if oracle_id is None:
                    oracle_id = _get_oracle_id_from_db(store, cid) or cid
                    oracle_cache[cid] = oracle_id

                image_id = None
                if img_path:
                    rel_path = _ensure_relative_path(img_path)
                    cache_key = (cid, rel_path)
                    image_id = image_cache.get(cache_key)
                    if image_id is None:
                        lang = _infer_language_from_filename(img_path)
                        image_id = store.get_or_create_image(
                            scryfall_id=cid,
                            oracle_id=oracle_id,
                            file_path=rel_path,
                            source="scryfall",
                            language=lang,
                            is_training=True,
                        )
                        image_cache[cache_key] = image_id

                idx = aug_index_counter.get(cid, 0)
                aug_index_counter[cid] = idx + 1

                store.add_embedding(
                    scryfall_id=cid,
                    oracle_id=oracle_id,
                    vec=vec_np,
                    mode=args.mode,
                    aug_index=idx,
                    image_id=image_id,
                )
                num_embeddings += 1

    print(f"[INFO] Nutze export_batch_size={export_batch_size} | num_workers={num_workers}")
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        futures = []
        for name, path in files:
            futures.append(
                executor.submit(
                    _prepare_card_tensors,
                    name,
                    path,
                    full_transform,
                    full_crop_cfg,
                    use_augmentations,
                    num_augmentations,
                    camera_aug_cfg,
                )
            )

        for future in tqdm(futures, desc="Vorbereiten", unit="card"):
            card_dict, tensors = future.result()
            cid = card_dict["card_uuid"]
            card_meta[cid] = card_dict
            for t in tensors:
                pending.append((t, cid, card_dict["image_path"]))
                if len(pending) >= export_batch_size:
                    _flush_pending()

        _flush_pending()

    card_order = [card_meta[cid]["card_uuid"] for cid in card_meta.keys()]
    total_cards = len(card_order)
    base_count = 1 + (num_augmentations if use_augmentations and num_augmentations > 0 else 0)
    expected_per_card = _expected_vectors_per_card(export_cfg, base_count)
    for cid in card_order:
        if cid not in centroid_sum:
            continue
        # compute centroid from accumulated sums/counts
        centroid = centroid_sum[cid] / float(max(1, centroid_count.get(cid, 1)))

        oracle_id = oracle_cache.get(cid)
        if oracle_id is None:
            oracle_id = _get_oracle_id_from_db(store, cid) or cid
            oracle_cache[cid] = oracle_id

        final_records: List[Tuple[np.ndarray, Optional[str], int]] = []
        if use_centroid:
            final_records = [(centroid, None, 999)]
        elif append_centroid:
            final_records = [(centroid, None, 999)]
        else:
            final_records = []

        for vec, path, aug_idx in final_records:
            store.add_embedding(
                scryfall_id=cid, oracle_id=oracle_id, vec=vec, mode=args.mode, aug_index=aug_idx, image_id=None
            )
            num_embeddings += 1

    storage_desc = "unveraendert"
    if use_centroid:
        storage_desc = "nur Zentroid"
    elif append_individual and append_centroid:
        storage_desc = "Einzel + Zentroid"
    elif append_individual:
        storage_desc = "nur Einzel-Embeddings"
    elif append_centroid:
        storage_desc = "nur Zentroid angehaengt"

    print(f"[SUMMARY] Modus: {args.mode} | Karten/Cluster: {total_cards} | Embeddings geschrieben: {num_embeddings}")
    print(f"[SUMMARY] Erwartet pro Karte: {expected_per_card} | Gesamt erwartet: {expected_per_card * total_cards}")
    print(f"[SUMMARY] Augmentierungen: {'aktiv' if use_augmentations and num_augmentations > 0 else 'aus'} (n={num_augmentations})")
    print(f"[SUMMARY] Speicherlogik: {storage_desc}")
    print(f"[NEXT] Erkennung testen: python -m src.recognize_cards --config {args.config}")


if __name__ == "__main__":
    main()
