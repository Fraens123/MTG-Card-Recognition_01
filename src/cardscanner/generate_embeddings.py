import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cardscanner.dataset import parse_scryfall_filename
from src.cardscanner.crop_utils import crop_set_symbol
import torch.utils.data as data

# PyTorch Dataset für parallele Embedding-Generierung
class CardAugmentDataset(data.Dataset):
    def __init__(
        self,
        image_files,
        augmentor,
        num_augmentations,
        transform,
        meta_parser,
        precomputed_variants=None,
        use_precomputed=False,
        precomputed_limit=None,
    ):
        self.image_files = image_files
        self.augmentor = augmentor
        self.num_augmentations = num_augmentations
        self.transform = transform
        self.meta_parser = meta_parser
        self.precomputed_variants = precomputed_variants or {}
        self.use_precomputed = use_precomputed and bool(self.precomputed_variants)
        self.precomputed_limit = precomputed_limit
        self.samples = []
        self._build_sample_list()

    def _build_sample_list(self):
        for img_path in self.image_files:
            meta = self.meta_parser(os.path.basename(img_path))
            if not meta:
                continue
            card_uuid = meta[0]
            # Stelle sicher, dass das Originalbild immer mindestens einmal eingebettet wird
            self.samples.append({
                "mode": "original",
                "orig_path": str(img_path),
                "meta": meta,
            })
            if self.use_precomputed and card_uuid in self.precomputed_variants:
                variants = self.precomputed_variants[card_uuid]
                if self.precomputed_limit:
                    variants = variants[: self.precomputed_limit]
                for full_path, symbol_path in variants:
                    self.samples.append({
                        "mode": "precomputed",
                        "full_path": full_path,
                        "symbol_path": symbol_path,
                        "meta": meta,
                        "orig_path": str(img_path),
                    })
            else:
                for aug_idx in range(1, self.num_augmentations + 1):
                    self.samples.append({
                        "mode": "augment",
                        "orig_path": str(img_path),
                        "meta": meta,
                        "aug_idx": aug_idx,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mode = sample["mode"]
        meta = sample["meta"]

        if mode == "precomputed":
            full_img = Image.open(sample["full_path"]).convert("RGB")
            symbol_path = sample.get("symbol_path")
            if symbol_path and os.path.exists(symbol_path):
                crop_img = Image.open(symbol_path).convert("RGB")
            else:
                crop_img = crop_set_symbol(full_img, None)
            img_path = sample["orig_path"]
        else:
            img = Image.open(sample["orig_path"]).convert("RGB")
            if mode == "augment" and self.augmentor is not None:
                full_img = self.augmentor.create_camera_like_augmentations(img, 1)[-1]
            else:
                full_img = img
            crop_img = crop_set_symbol(full_img, None)
            img_path = sample["orig_path"]

        tensor_full = self.transform(full_img)
        tensor_crop = self.transform(crop_img)
        return (tensor_full, tensor_crop), img_path, meta


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


def load_precomputed_variants(precomputed_root: str) -> Dict[str, List[Tuple[str, Optional[str]]]]:
    """
    Liest vorberechnete Augmentierungen aus data/precomputed_augmented/…
    und liefert ein Mapping card_uuid -> [(full_path, symbol_path), ...]
    """
    variant_map: Dict[str, List[tuple]] = {}
    if not precomputed_root or not os.path.isdir(precomputed_root):
        return variant_map

    for dirpath, dirnames, _ in os.walk(precomputed_root):
        if os.path.basename(dirpath) != "full":
            continue
        card_dir = os.path.dirname(dirpath)
        card_uuid = os.path.basename(card_dir)
        symbol_dir = os.path.join(card_dir, "symbol")
        variants = []
        for fname in sorted(os.listdir(dirpath)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            full_path = os.path.join(dirpath, fname)
            symbol_path = None
            candidate = os.path.join(symbol_dir, fname)
            if os.path.exists(candidate):
                symbol_path = candidate
            variants.append((full_path, symbol_path))
        if variants:
            variant_map[card_uuid] = variants
    return variant_map
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.cardscanner.augment_cards import CameraLikeAugmentor
#!/usr/bin/env python3
"""
Script zur Generierung von Embeddings für alle Karten mit dem trainierten Modell.
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

# Absolute Imports für direkten Aufruf
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.cardscanner.model import load_encoder
from src.cardscanner.db import SimpleCardDB
from src.cardscanner.dataset import parse_scryfall_filename


def load_config(config_path: str = "config.yaml") -> dict:
    """Lädt YAML-Konfiguration direkt"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config-Datei nicht gefunden: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise ValueError(f"Config-Datei ist leer: {config_path}")
    
    return config


def get_inference_transforms(resize_hw: tuple = (320, 224)) -> T.Compose:
    """
    Transform-Pipeline für Inference (ohne Augmentierung). resize_hw = (H, W).
    """
    return T.Compose([
        T.Resize(resize_hw, antialias=True),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def generate_embeddings_from_directory(config: dict, model_path: str, images_dir: str, 
                                     resize_hw: tuple, device: torch.device,
                                     backup_db: bool = True) -> Dict[str, any]:
    """
    Generiert Embeddings für alle Bilder in einem Verzeichnis
    """
    embed_dim = config['model']['embed_dim']
    print(f"🔧 Lade trainiertes Modell: {model_path} (embed_dim={embed_dim})")
    model = load_encoder(model_path, embed_dim=embed_dim, device=device)
    model.eval()
    model.to(device)
    
    print(f"📂 Verarbeite Bilder aus: {images_dir}")
    
    # Backup der bestehenden Database
    if backup_db:
        config = load_config()
        db_path = config['database']['path']
        if os.path.exists(db_path):
            backup_path = f"{db_path}.backup"
            shutil.copy2(db_path, backup_path)
            print(f"💾 Database-Backup erstellt: {backup_path}")
    
    # Neue Database erstellen (überschreibt die alte!)
    db = SimpleCardDB()

    # Transform für Inference
    transform = get_inference_transforms(resize_hw)
    emb_export = config.get("embedding_export", {})
    num_augmentations = emb_export.get("num_augmentations", 20)
    batch_size = emb_export.get("batch_size", 64)
    workers = emb_export.get("workers", 4)
    use_precomputed = emb_export.get("use_precomputed", False)
    precomputed_limit = emb_export.get("precomputed_limit")
    if precomputed_limit is not None:
        try:
            precomputed_limit = int(precomputed_limit)
        except (TypeError, ValueError):
            print(f"[WARN] Ungültiger Wert für embedding_export.precomputed_limit -> {precomputed_limit}. Verwende alle Varianten.")
            precomputed_limit = None
    data_cfg = config.get("data", {})
    precomputed_root = data_cfg.get("precomputed_augmented")
    precomputed_variants = {}
    if use_precomputed:
        precomputed_variants = load_precomputed_variants(precomputed_root)
        total_precomputed = sum(len(v) for v in precomputed_variants.values())
        if not precomputed_variants:
            print("[INFO] Keine vorberechneten Augmentierungen gefunden – fallback auf On-the-fly.")
            use_precomputed = False
        else:
            print(f"[INFO] Nutze vorberechnete Augmentierungen: {len(precomputed_variants)} Karten / {total_precomputed} Varianten")
            if precomputed_limit:
                print(f"[INFO] Max. Varianten pro Karte für Export: {precomputed_limit}")

    aug_params = config.get("augmentation", {})
    camera_augmentor = None
    if not use_precomputed:
        camera_augmentor = CameraLikeAugmentor(
            brightness_range=(aug_params.get("brightness_min", 0.9), aug_params.get("brightness_max", 1.3)),
            contrast_range=(aug_params.get("contrast_min", 0.98), aug_params.get("contrast_max", 1.02)),
            blur_range=(0.0, aug_params.get("blur_max", 3.2)),
            noise_range=(0, aug_params.get("noise_max", 5.0)),
            rotation_range=(-aug_params.get("rotation_max", 5.0), aug_params.get("rotation_max", 5.0)),
            perspective=aug_params.get("perspective", 0.05),
            shadow=aug_params.get("shadow", 0.14),
            saturation_range=(aug_params.get("saturation_min", 0.8), aug_params.get("saturation_max", 1.2)),
            color_temperature_range=(aug_params.get("color_temperature_min", 0.84), aug_params.get("color_temperature_max", 1.16)),
            hue_shift_max=aug_params.get("hue_shift_max", 15.0),
            background_color=aug_params.get("background_color", "white"),
        )

    # Alle Bilddateien finden
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(images_dir).glob(ext))

    print(f"\n[INFO] Gefundene Kartenbilder: {len(image_files)}")
    # Zähle eindeutige Karten
    unique_cards = set()
    for img_file in image_files:
        meta = parse_scryfall_filename(img_file.name)
        if meta:
            card_uuid, set_code, collector_number, card_name = meta
            unique_cards.add(f"{set_code}_{collector_number}_{card_name}")
    print(f"[INFO] Eindeutige Karten: {len(unique_cards)}")
    print(f"[INFO] Embeddings/Centroids pro Karte (max): {config['database'].get('n_centroids', 12)}")
    print(f"[INFO] Erwartete Gesamtzahl Centroids: {len(unique_cards) * config['database'].get('n_centroids', 12)}")

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
        camera_augmentor,
        num_augmentations,
        transform,
        parse_scryfall_filename,
        precomputed_variants=precomputed_variants,
        use_precomputed=use_precomputed,
        precomputed_limit=precomputed_limit,
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
                emb_full = model(batch_full)
                emb_full = nn.functional.normalize(emb_full, p=2, dim=-1)

                emb_crop = model(batch_crop)
                emb_crop = nn.functional.normalize(emb_crop, p=2, dim=-1)

                # Concat Full + Crop, dann nochmal normalisieren
                embeddings = torch.cat([emb_full, emb_crop], dim=-1)
                embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)

                embeddings_np = embeddings.cpu().numpy()

            for i, embedding_np in enumerate(embeddings_np):
                meta = metas[i]
                if not meta or len(meta) != 4:
                    print(f"⚠️  Kann nicht parsen: {img_paths[i]}")
                    errors += 1
                    continue
                card_uuid, set_code, collector_number, card_name = meta
                unique_id = f"{set_code}_{collector_number}_{card_name}"
                card_meta[unique_id] = (card_uuid, set_code, collector_number, card_name, img_paths[i])
                card_embeddings[unique_id].append(embedding_np)
        except Exception as e:
            print(f"❌ Fehler im Batch: {e}")
            errors += 1
            continue

    # K-Means-Centroids berechnen und speichern (unverändert, aber mit neuen card_embeddings/card_meta)
    from sklearn.cluster import KMeans
    n_centroids = int(config["database"].get("n_centroids", 12))
    single_cluster_cards = 0
    multi_cluster_cards = 0
    for unique_id, emb_list in card_embeddings.items():
        if unique_id not in card_meta:
            print(f"[WARN] Keine Metadaten für {unique_id}")
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
                embedding=centroid
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
                    embedding=centroid
                )
            multi_cluster_cards += 1

    db.save_to_file()
    
    print(f"\n✅ Embedding-Generierung abgeschlossen!")
    total_centroids = sum(n_centroids if len(v) >= n_centroids else 1 for v in card_embeddings.values())
    print(f"📊 Generiert: {total_centroids} Centroids")
    print(f"❌ Fehler: {errors}")
    print(f"💾 Database gespeichert: {db.db_path}")
    # Statistik-Ausgabe
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
        'database_path': str(db.db_path)
    }


def main():
    """
    Hauptfunktion für Embedding-Generierung - alle Parameter aus config.yaml
    """
    # Config laden
    try:
        config = load_config("config.yaml")
        print(f"📋 Config geladen aus: config.yaml")
    except Exception as e:
        print(f"❌ Fehler beim Laden der Config: {e}")
        return
    
    # Parameter aus Config
    model_path = config['model']['weights_path']
    
    # Embedding-Modus aus Config lesen
    embedding_mode = config['database'].get('embedding_mode', 'original')
    
    # Bildverzeichnis basierend auf Modus wählen
    if embedding_mode == 'augmented':
        images_dir = config['data']['scryfall_augmented']
        print(f"🔄 Modus: AUGMENTED - Mehr Variationen, langsamere Suche")
        print(f"📊 Geschätzte Embeddings: ~1,600 (mehr Speicher, langsamere Suche)")
        print(f"⚠️  EXPERIMENTELLER MODUS: Nicht empfohlen - keine bessere Erkennungsqualität")
    else:
        images_dir = config['data']['scryfall_images']
        print(f"⚡ Modus: ORIGINAL - Schneller, weniger Speicherbedarf (EMPFOHLEN)")
        print(f"📊 Geschätzte Embeddings: ~400 (weniger Speicher, schnellere Suche)")
        print(f"✅ Tests zeigen identische Erkennungsqualität bei besserer Performance!")
    
    backup_db = True  # Immer Backup erstellen
    
    # Device
    hardware_config = config['hardware']
    cuda_available = torch.cuda.is_available()
    use_cuda = hardware_config['use_cuda'] and cuda_available
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print(f"🚀 MTG Card Embedding-Generierung")
    print(f"🖥️  Device: {device.type.upper()}")
    if cuda_available and use_cuda:
        print(f"   GPU: {torch.cuda.get_device_name()}")
    elif cuda_available and not use_cuda:
        print(f"   CUDA verfügbar, aber deaktiviert")
    else:
        print(f"   CUDA nicht verfügbar")
    
    print(f"📂 Bilder-Verzeichnis: {images_dir}")
    print(f"🔧 Modell: {model_path}")
    
    # Prüfungen
    if not os.path.exists(model_path):
        print(f"❌ Modell nicht gefunden: {model_path}")
        print("   Bitte zuerst Training durchführen!")
        return
    
    if not os.path.exists(images_dir):
        print(f"❌ Bilder-Verzeichnis nicht gefunden: {images_dir}")
        return
    
    # Bildgröße automatisch erkennen (wie beim Training)
    if config['training']['auto_detect_size']:
        from src.cardscanner.train_triplet import detect_image_size
        target_width, target_height = detect_image_size(images_dir)
    else:
        target_width = config['training']['target_width']
        target_height = config['training']['target_height']
    
    resize_hw = (target_height, target_width)
    print(f"📐 Bildgröße (Breite x Höhe): {target_width}x{target_height}")
    
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
        print(f"\n🎉 Embedding-Generierung erfolgreich!")
        print(f" Embeddings generiert: {result['embeddings_generated']}")
        if result['errors'] > 0:
            print(f"⚠️  Fehler beim Parsen: {result['errors']} Dateien")
        print(f"💿 Database: {result['database_path']}")
        print("➡️  Nächster Schritt: Kartenerkennung testen mit recognize_cards.py")
    else:
        print(f"\n❌ Embedding-Generierung fehlgeschlagen!")


if __name__ == "__main__":
    main()
