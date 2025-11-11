import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch.utils.data as data

from src.cardscanner.dataset import parse_scryfall_filename, crop_set_symbol
from src.cardscanner.embedding_utils import build_card_embedding
from src.cardscanner.image_pipeline import (
    build_resize_normalize_transform,
    get_set_symbol_crop_cfg,
    resolve_resize_hw,
)
from torchvision import transforms as tv_transforms
from torchvision.transforms.functional import InterpolationMode


def build_export_augmentation_transform() -> tv_transforms.Compose:
    """Torch-basierte Augmentierung ??hnlich der Trainingsvariante (aber leichter)."""
    return tv_transforms.Compose(
        [
            tv_transforms.RandomApply(
                [tv_transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02)],
                p=0.8,
            ),
            tv_transforms.RandomApply([tv_transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.7))], p=0.3),
            tv_transforms.RandomRotation(
                degrees=2,
                interpolation=InterpolationMode.BICUBIC,
                fill=0,
                expand=False,
            ),
        ]
    )

# PyTorch Dataset f??r parallele Embedding-Generierung
class CardAugmentDataset(data.Dataset):
    def __init__(
        self,
        image_files,
        full_transform,
        crop_transform,
        meta_parser,
        set_symbol_crop_cfg,
        num_variants: int,
        camera_augmentor=None,
        enable_camera_aug: bool = False,
    ):
        self.image_files = image_files
        self.full_transform = full_transform
        self.crop_transform = crop_transform
        self.meta_parser = meta_parser
        self.set_symbol_crop_cfg = set_symbol_crop_cfg
        self.total_variants = max(num_variants, 1)
        self.camera_augmentor = camera_augmentor
        self.use_camera_augmentor = bool(enable_camera_aug and camera_augmentor is not None)
        self.random_augment = build_export_augmentation_transform()
        self.samples = []
        for img_path in self.image_files:
            for variant_idx in range(self.total_variants):
                self.samples.append((img_path, variant_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, variant_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        aug_img = img
        if variant_idx > 0:
            if self.use_camera_augmentor:
                aug_img = self.camera_augmentor.create_camera_like_augmentations(img, 1)[-1]
            else:
                aug_img = self.random_augment(img)

        # Pipeline entspricht dem Training: Bild -> (optional) Kamera-Augmentor -> Crop -> Resize/ToTensor/Normalize.
        tensor_full = self.full_transform(aug_img)

        # Set-Symbol-Crop (aus augmentiertem Bild)
        crop_img = crop_set_symbol(aug_img, self.set_symbol_crop_cfg)
        tensor_crop = self.crop_transform(crop_img)

        import os
        meta = self.meta_parser(os.path.basename(img_path))
        # R??ckgabe: ((full, crop), pfad, meta)
        return (tensor_full, tensor_crop), str(img_path), meta


def collate_card_batch(batch):
    """
    Custom collate_fn, damit Meta-Informationen pro Sample erhalten bleiben.
    """
    full_list = []
    crop_list = []
    paths = []
    metas = []

    for (t_full, t_crop), path, meta in batch:
        full_list.append(t_full)
        crop_list.append(t_crop)
        paths.append(path)
        metas.append(meta)

    full_batch = torch.stack(full_list, dim=0)
    crop_batch = torch.stack(crop_list, dim=0)
    return (full_batch, crop_batch), paths, metas
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.cardscanner.augment_cards import CameraLikeAugmentor
#!/usr/bin/env python3
"""
Script zur Generierung von Embeddings f??r alle Karten mit dem trainierten Modell.
Speichert Embeddings in JSON-Database.
"""

import os
import yaml
import sys
import shutil
from pathlib import Path
from typing import Dict, List
import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm

# Absolute Imports f??r direkten Aufruf
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.cardscanner.model import load_encoder
from src.cardscanner.db import SimpleCardDB


def load_config(config_path: str = "config.yaml") -> dict:
    """L??dt YAML-Konfiguration direkt"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config-Datei nicht gefunden: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise ValueError(f"Config-Datei ist leer: {config_path}")
    
    return config


def get_inference_transforms(resize_hw: tuple = (320, 224)) -> T.Compose:
    """
    Transform-Pipeline f??r Inference (ohne Augmentierung). resize_hw = (H, W).
    """
    return build_resize_normalize_transform(resize_hw)


def generate_embeddings_from_directory(config: dict, model_path: str, images_dir: str, 
                                     resize_hw: tuple, device: torch.device,
                                     backup_db: bool = True) -> Dict[str, any]:
    """
    Generiert Embeddings f??r alle Bilder in einem Verzeichnis
    """
    embed_dim = config['model']['embed_dim']
    print(f"[TOOL] Lade trainiertes Modell: {model_path} (embed_dim={embed_dim})")
    model = load_encoder(model_path, embed_dim=embed_dim, device=device)
    model.eval()
    model.to(device)
    
    print(f"[DIR] Verarbeite Bilder aus: {images_dir}")
    
    # Backup der bestehenden Database
    if backup_db:
        config = load_config()
        db_path = config['database']['path']
        if os.path.exists(db_path):
            backup_path = f"{db_path}.backup"
            shutil.copy2(db_path, backup_path)
            print(f"[SAVE] Database-Backup erstellt: {backup_path}")
    
    # Neue Database erstellen (??berschreibt die alte!)
    db = SimpleCardDB()

    # Transform f?r Inference
    full_transform = get_inference_transforms(resize_hw)
    emb_export = config.get("embedding_export", {})
    use_original_mode = emb_export.get("use_original_mode", True)
    num_variants = max(int(emb_export.get("num_augmentations", 1)), 1)
    batch_size = emb_export.get("batch_size", 64)
    workers = emb_export.get("workers", 4)
    aug_params = config.get("camera_augmentor", config.get("augmentation", {}))
    set_symbol_crop_cfg = get_set_symbol_crop_cfg(config) or {}
    crop_resize_hw = (set_symbol_crop_cfg.get("target_height", 64), set_symbol_crop_cfg.get("target_width", 160))
    crop_transform = build_resize_normalize_transform(crop_resize_hw)
    camera_augmentor = None
    raw_camera_flag = emb_export.get("use_camera_augmentor", False)
    effective_camera_aug = raw_camera_flag and use_original_mode
    if effective_camera_aug:
        camera_augmentor = CameraLikeAugmentor(**aug_params)
        print("[INFO] Embedding-Export nutzt CameraLikeAugmentor (Original-Modus).")
    else:
        if raw_camera_flag and not use_original_mode:
            print("[INFO] Kamera-Augmentor im Vollmodus deaktiviert (Torch-Augmentierungen ?bernehmen).")
        print("[INFO] Embedding-Export ohne CameraLikeAugmentor (Torch-Augmentierungen + Resize).")


    # Alle Bilddateien finden
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(images_dir).glob(ext))

    print(f"\n[INFO] Gefundene Kartenbilder: {len(image_files)}")
    # Z??hle eindeutige Karten
    unique_cards = set()
    for img_file in image_files:
        meta = parse_scryfall_filename(img_file.name)
        if meta:
            card_uuid, set_code, collector_number, card_name = meta
            unique_cards.add(f"{set_code}_{collector_number}_{card_name}")
    print(f"[INFO] Eindeutige Karten: {len(unique_cards)}")
    if use_original_mode:
        max_centroids = config["database"].get("n_centroids", 12)
        print(f"[INFO] Export-Modus: Original -> KMeans (max {max_centroids} Centroids pro Karte)")
        print(f"[INFO] Erwartete Gesamtzahl Centroids: {len(unique_cards) * max_centroids}")
    else:
        expected_embeddings = len(unique_cards) * num_variants
        print(f"[INFO] Export-Modus: Voll -> {num_variants} Augmentierungen pro Karte")
        print(f"[INFO] Erwartete Gesamtzahl Embeddings: {expected_embeddings}")

    if not image_files:
        print(f"[WARN] Keine Bilder gefunden in: {images_dir}")
        return {
            'embeddings_generated': 0,
            'errors': 0,
            'database_path': str(SimpleCardDB().db_path)
        }
    # Embeddings pro Karte sammeln (parallele Verarbeitung mit DataLoader)
    from collections import defaultdict
    card_embeddings = defaultdict(list)
    card_meta = dict()
    errors = 0

    dataset = CardAugmentDataset(
        image_files,
        full_transform,
        crop_transform,
        parse_scryfall_filename,
        set_symbol_crop_cfg,
        num_variants,
        camera_augmentor=camera_augmentor,
        enable_camera_aug=effective_camera_aug,
    )
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_card_batch,
    )

    for (batch_full, batch_crop), img_paths, metas in tqdm(loader, desc="Embeddings", unit="batch"):
        try:
            batch_full = batch_full.to(device, non_blocking=True)
            batch_crop = batch_crop.to(device, non_blocking=True)
            with torch.no_grad():
                embeddings = build_card_embedding(model, batch_full, batch_crop)
                embeddings_np = embeddings.cpu().numpy()

            for i, embedding_np in enumerate(embeddings_np):
                meta = metas[i]
                if not meta or len(meta) != 4:
                    print(f"[WARN]  Kann nicht parsen: {img_paths[i]}")
                    errors += 1
                    continue
                card_uuid, set_code, collector_number, card_name = meta
                unique_id = f"{set_code}_{collector_number}_{card_name}"
                card_meta[unique_id] = (card_uuid, set_code, collector_number, card_name, img_paths[i])
                card_embeddings[unique_id].append(embedding_np)
        except Exception as e:
            print(f"[ERROR] Fehler im Batch: {e}")
            errors += 1
            continue

    if use_original_mode:
        from sklearn.cluster import KMeans
        n_centroids = int(config["database"].get("n_centroids", 12))
        single_cluster_cards = 0
        multi_cluster_cards = 0
        for unique_id, emb_list in card_embeddings.items():
            if unique_id not in card_meta:
                print(f"[WARN] Keine Metadaten fOr {unique_id}")
                continue
            card_uuid, set_code, collector_number, card_name, img_file = card_meta[unique_id]
            display_name = f"{card_name.replace('_', ' ')} ({set_code})"
            vectors = np.stack(emb_list)
            if len(emb_list) < max(3, n_centroids):
                centroid = np.mean(vectors, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
                db.add_card(
                    card_uuid=card_uuid,
                    name=display_name,
                    set_code=set_code,
                    collector_number=collector_number,
                    image_path=str(img_file),
                    embedding=centroid,
                )
                single_cluster_cards += 1
            else:
                kmeans = KMeans(n_clusters=n_centroids, random_state=42, n_init='auto')
                kmeans.fit(vectors)
                centroids = kmeans.cluster_centers_
                centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
                for i, centroid in enumerate(centroids):
                    db.add_card(
                        card_uuid=f"{card_uuid}_c{i}",
                        name=display_name,
                        set_code=set_code,
                        collector_number=collector_number,
                        image_path=str(img_file),
                        embedding=centroid,
                    )
                multi_cluster_cards += 1

        db.save_to_file()
        print("\n[OK] Embedding-Generierung abgeschlossen!")
        total_centroids = sum(n_centroids if len(v) >= n_centroids else 1 for v in card_embeddings.values())
        print(f"[INFO] Generiert: {total_centroids} Centroids")
        print(f"[INFO] Fehler: {errors}")
        print(f"[INFO] Database gespeichert: {db.db_path}")
        total_datapoints = sum(len(embeddings) for embeddings in card_embeddings.values())
        total_clusters = sum(n_centroids if len(embeddings) >= n_centroids else 1 for embeddings in card_embeddings.values())
        print(f"[INFO] Embedding-Export abgeschlossen.")
        print(f"[INFO] Erzeugte Cluster (Centroids): {total_clusters}")
        print(f"[INFO] Gesamtzahl der Embedding-Datenpunkte: {total_datapoints}")
        print(f"[INFO] Karten mit 1 Cluster (<{n_centroids} Embeddings): {single_cluster_cards}")
        print(f"[INFO] Karten mit {n_centroids} Clustern: {multi_cluster_cards}")
        return {
            'embeddings_generated': total_centroids,
            'errors': errors,
            'database_path': str(db.db_path),
        }

    saved_embeddings = 0
    cards_with_multiple = 0
    for unique_id, emb_list in card_embeddings.items():
        if unique_id not in card_meta:
            print(f"[WARN] Keine Metadaten fOr {unique_id}")
            continue
        card_uuid, set_code, collector_number, card_name, img_file = card_meta[unique_id]
        display_name = f"{card_name.replace('_', ' ')} ({set_code})"
        vectors = np.stack(emb_list)
        if len(emb_list) > 1:
            cards_with_multiple += 1
        for i, vector in enumerate(vectors):
            db.add_card(
                card_uuid=f"{card_uuid}_a{i}",
                name=display_name,
                set_code=set_code,
                collector_number=collector_number,
                image_path=str(img_file),
                embedding=vector,
            )
        saved_embeddings += len(emb_list)

    db.save_to_file()
    print("\n[OK] Embedding-Generierung abgeschlossen!")
    print(f"[INFO] Gespeicherte Embeddings: {saved_embeddings}")
    print(f"[INFO] Karten mit >1 Embedding: {cards_with_multiple}")
    print(f"[INFO] Fehler: {errors}")
    print(f"[INFO] Database gespeichert: {db.db_path}")
    return {
        'embeddings_generated': saved_embeddings,
        'errors': errors,
        'database_path': str(db.db_path),
    }



def main():
    """
    Hauptfunktion f??r Embedding-Generierung - alle Parameter aus config.yaml
    """
    # Config laden
    try:
        config = load_config("config.yaml")
        print("Config geladen aus: config.yaml")
    except Exception as e:
        print(f"[ERROR] Fehler beim Laden der Config: {e}")
        return
    
    # Parameter aus Config
    model_path = config['model']['weights_path']
    
    # Embedding-Modus aus Config lesen
    embedding_mode = config['database'].get('embedding_mode', 'original')
    
    # Bildverzeichnis basierend auf Modus w??hlen
    if embedding_mode == 'augmented':
        images_dir = config['data']['scryfall_augmented']
        print(f"[REFRESH] Modus: AUGMENTED - Mehr Variationen, langsamere Suche")
        print(f"[STATS] Gesch??tzte Embeddings: ~1,600 (mehr Speicher, langsamere Suche)")
        print(f"[WARN]  EXPERIMENTELLER MODUS: Nicht empfohlen - keine bessere Erkennungsqualit??t")
    else:
        images_dir = config['data']['scryfall_images']
        print(f"[FAST] Modus: ORIGINAL - Schneller, weniger Speicherbedarf (EMPFOHLEN)")
        print(f"[STATS] Gesch??tzte Embeddings: ~400 (weniger Speicher, schnellere Suche)")
        print(f"[OK] Tests zeigen identische Erkennungsqualit??t bei besserer Performance!")
    
    backup_db = True  # Immer Backup erstellen
    
    # Device
    hardware_config = config['hardware']
    cuda_available = torch.cuda.is_available()
    use_cuda = hardware_config['use_cuda'] and cuda_available
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print(f"[GO] MTG Card Embedding-Generierung")
    print(f"[SCREEN]  Device: {device.type.upper()}")
    if cuda_available and use_cuda:
        print(f"   GPU: {torch.cuda.get_device_name()}")
    elif cuda_available and not use_cuda:
        print(f"   CUDA verf??gbar, aber deaktiviert")
    else:
        print(f"   CUDA nicht verf??gbar")
    
    print(f"[DIR] Bilder-Verzeichnis: {images_dir}")
    print(f"[TOOL] Modell: {model_path}")
    
    # Pr??fungen
    if not os.path.exists(model_path):
        print(f"[ERROR] Modell nicht gefunden: {model_path}")
        print("   Bitte zuerst Training durchf??hren!")
        return
    
    if not os.path.exists(images_dir):
        print(f"[ERROR] Bilder-Verzeichnis nicht gefunden: {images_dir}")
        return
    
    # Bildgr??e identisch zum Training bestimmen
    resize_hw = resolve_resize_hw(config, images_dir)
    target_height, target_width = resize_hw
    print(f"[INFO] Embedding-Export resize (W x H): {target_width}x{target_height}")

    
    # Embeddings generieren
    result = generate_embeddings_from_directory(
        config=config,
        model_path=model_path,
        images_dir=images_dir,
        resize_hw=resize_hw,
        device=device,
        backup_db=backup_db
    )
    
    # Ergebnisse
    if result['embeddings_generated'] > 0:
        print(f"\n[DONE] Embedding-Generierung erfolgreich!")
        print(f" Embeddings generiert: {result['embeddings_generated']}")
        if result['errors'] > 0:
            print(f"[WARN]  Fehler beim Parsen: {result['errors']} Dateien")
        print(f"[DISK] Database: {result['database_path']}")
        print("->  N??chster Schritt: Kartenerkennung testen mit recognize_cards.py")
    else:
        print(f"\n[ERROR] Embedding-Generierung fehlgeschlagen!")


if __name__ == "__main__":
    main()

