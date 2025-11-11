from collections import OrderedDict
import os
import random
from typing import Dict, List, Tuple, Optional

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.utils import save_image

from src.cardscanner.augment_cards import CameraLikeAugmentor
from src.cardscanner.image_pipeline import crop_set_symbol, get_set_symbol_crop_cfg, build_resize_normalize_transform



_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _denormalize_for_save(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    img = tensor * _STD + _MEAN
    return torch.clamp(img, 0.0, 1.0)


def parse_scryfall_filename(filename: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parst Scryfall-Dateinamen in verschiedenen Formaten:
    Format 1: {set_code}_{collector_number}_{card_name}_{card_uuid} (z.B. 10E_31_Pacifism_686352f6.jpg)
    Format 2: {set_code}-{collector_number}-{card_name} (z.B. war-65-rescuer-sphinx.png)
    Gibt zur-ck: (card_uuid, set_code, collector_number, card_name)
    """
    name, _ = os.path.splitext(os.path.basename(filename))

    # Format 1: {set_code}_{collector_number}_{card_name}_{card_uuid}
    parts = name.split("_")
    if len(parts) >= 4:
        set_code = parts[0]
        collector_number = parts[1]
        card_uuid = parts[-1]
        card_name = "_".join(parts[2:-1])
        return card_uuid, set_code, collector_number, card_name

    # Format 2: {set_code}-{collector_number}-{card_name}
    if "-" in name:
        parts = name.split("-", 2)
        if len(parts) == 3:
            set_code = parts[0]
            collector_number = parts[1]
            card_name = parts[2]
            card_uuid = f"{set_code}_{collector_number}_{hash(name) % 100000:05d}"
            return card_uuid, set_code, collector_number, card_name

    # Robust: Wenn Format nicht passt, logge Warnung und gib None zur-ck
    print(f"[WARN] parse_scryfall_filename: Unbekanntes Format f-r '{filename}'")
    return None


class TripletImageDataset(Dataset):
    def __init__(
        self,
        scryfall_dir: str,
        camera_dir: Optional[str],
        transform_anchor: T.Compose = None,
        transform_posneg: T.Compose = None,
        transform_crop: T.Compose = None,
        seed: int = 42,
        use_camera_augmentor: bool = True,
        augmentor_params: dict = None,
        cache_images: bool = True,
        cache_max_size: int = 1000,
        max_samples_per_epoch: Optional[int] = None,
        set_symbol_crop_cfg: Optional[dict] = None,
    ) -> None:
        """
        Dataset f-r MTG-Karten Triplet Training.
        Verwendet scryfall_dir als Quelle f-r augmentierte Bilder.
        """
        random.seed(seed)
        self.scryfall_dir = scryfall_dir  # Verzeichnis mit Originalbildern
        self.camera_dir = camera_dir
        self.transform_anchor = transform_anchor
        self.transform_posneg = transform_posneg
        self.use_camera_augmentor = use_camera_augmentor
        self.transform_crop = transform_crop

        # Set-Symbol-Crop einmalig bestimmen, damit Training/Export identisch schneiden.
        self.set_symbol_crop_cfg = set_symbol_crop_cfg
        if self.set_symbol_crop_cfg is None:
            try:
                with open("config.yaml", "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                self.set_symbol_crop_cfg = get_set_symbol_crop_cfg(cfg)
            except Exception as e:
                print(f"[WARN] TripletImageDataset: config.yaml nicht lesbar: {e}")
                self.set_symbol_crop_cfg = None
        crop_target_h = 64
        crop_target_w = 160
        if self.set_symbol_crop_cfg:
            crop_target_w = self.set_symbol_crop_cfg.get("target_width", crop_target_w)
            crop_target_h = self.set_symbol_crop_cfg.get("target_height", crop_target_h)
        if self.transform_crop is None:
            self.transform_crop = build_resize_normalize_transform((crop_target_h, crop_target_w))
        if self.use_camera_augmentor:
            if augmentor_params is None:
                with open("config.yaml", "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                augmentor_params = config.get("camera_augmentor", config.get("augmentation", {}))
            self.camera_augmentor = CameraLikeAugmentor(**augmentor_params)
        self.card_to_paths = {}
        if cache_images:
            self._image_cache = LRUCache(cache_max_size)
        else:
            self._image_cache = None
        self.cache_images = cache_images
        self.max_samples_per_epoch = max_samples_per_epoch

        print(f"Lade Originalbilder aus: {self.scryfall_dir}")

        def iter_top_level_images(folder: str):
            try:
                entries = os.listdir(folder)
            except FileNotFoundError:
                print(f"[WARN] Verzeichnis nicht gefunden: {folder}")
                return
            for name in entries:
                path = os.path.join(folder, name)
                if not os.path.isfile(path):
                    continue  # Unterordner werden bewusst ignoriert
                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                yield path, name

        # Lade nur die Dateien direkt im scryfall_dir (keine Unterordner)
        for img_path, filename in iter_top_level_images(self.scryfall_dir):
            meta = parse_scryfall_filename(filename)
            if not meta:
                print(f"Kann nicht parsen: {filename}")
                continue
            card_uuid = meta[0]
            self.card_to_paths.setdefault(card_uuid, []).append(img_path)
            if self.cache_images and len(self._image_cache) < self._image_cache.max_size:
                try:
                    self._image_cache[img_path] = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"[Cache] Fehler beim Laden von {img_path}: {e}")

        # Optional: Lade echte Kamera-Bilder
        if self.camera_dir and os.path.exists(self.camera_dir):
            print(f"Lade Camera-Bilder aus: {self.camera_dir}")
            for img_path, filename in iter_top_level_images(self.camera_dir):
                meta = parse_scryfall_filename(filename)
                if not meta:
                    continue
                card_uuid = meta[0]
                self.card_to_paths.setdefault(card_uuid, []).append(img_path)
                if self.cache_images and len(self._image_cache) < self._image_cache.max_size:
                    try:
                        self._image_cache[img_path] = Image.open(img_path).convert("RGB")
                    except Exception as e:
                        print(f"[Cache] Fehler beim Laden von {img_path}: {e}")

        self.card_ids = sorted(list(self.card_to_paths.keys()))
        self.card_to_idx = {card_id: idx for idx, card_id in enumerate(self.card_ids)}
        self.all_paths = [p for paths in self.card_to_paths.values() for p in paths]
        # Basismenge = alle vorhandenen Bildpfade (Scryfall + Kamera)
        self.total_available_samples = len(self.all_paths)
        if self.total_available_samples == 0:
            print("[WARN] TripletImageDataset: keine Trainingsbilder gefunden - Dataset bleibt leer.")

        print(f"Gefunden:")
        print(f"  - {len(self.card_ids)} eindeutige Karten")
        print(f"  - {len(self.all_paths)} Bilder insgesamt")
        for card_id in self.card_ids[:3]:  # Erste 3 als Beispiel
            print(f"  - Karte {card_id}: {len(self.card_to_paths[card_id])} Bilder")
        if len(self.card_ids) > 3:
            print("  - ...")
    def save_augmentations_of_first_card(self, output_dir: str, n_aug: int = 20):
        """
        Erzeugt n_aug augmentierte Varianten der ersten Karte und speichert sie.
        Zus-tzlich werden dazugeh-rige Set-Symbol-Crops gespeichert.
        Struktur:
          output_dir/
            full/
              <cardid>_aug_00.jpg
              ...
            symbol/
              <cardid>_aug_00.jpg
              ...
        """
        import os

        full_dir = os.path.join(output_dir, "full")
        symbol_dir = os.path.join(output_dir, "symbol")
        os.makedirs(full_dir, exist_ok=True)
        os.makedirs(symbol_dir, exist_ok=True)

        if not self.card_ids:
            print("Keine Karten im Dataset gefunden.")
            return

        card_id = self.card_ids[0]
        img_path = self.card_to_paths[card_id][0]
        img = Image.open(img_path).convert("RGB")
        print(f"Erzeuge {n_aug} Augmentierungen f-r Karte: {card_id} ({img_path})")

        if self.use_camera_augmentor:
            augmentations = self.camera_augmentor.create_camera_like_augmentations(
                img, num_augmentations=n_aug
            )
            # augmentations[0] ist Original, danach Augmentierungen
            for i, aug_img in enumerate(augmentations[:n_aug]):
                full_path = os.path.join(full_dir, f"{card_id}_aug_{i:02d}.jpg")
                aug_img.save(full_path, "JPEG", quality=85)

                # Symbol-Crop aus augmentiertem Bild
                crop_img = crop_set_symbol(aug_img, self.set_symbol_crop_cfg)
                symbol_path = os.path.join(symbol_dir, f"{card_id}_aug_{i:02d}.jpg")
                crop_img.save(symbol_path, "JPEG", quality=85)
        else:
            # Fallback: nur Transform-Anchor ohne Kameraaugmentor
            for i in range(n_aug):
                aug_tensor = self.transform_anchor(img) if self.transform_anchor else T.ToTensor()(img)
                full_path = os.path.join(full_dir, f"{card_id}_aug_{i:02d}.png")
                save_image(_denormalize_for_save(aug_tensor), full_path)

                crop_img = crop_set_symbol(img, self.set_symbol_crop_cfg)
                symbol_path = os.path.join(symbol_dir, f"{card_id}_aug_{i:02d}.png")
                crop_img.save(symbol_path, "PNG")

        print(f"Augmentierungen gespeichert in: {output_dir}")

    def __len__(self) -> int:
        """
        Basis-L-nge = Anzahl der real vorhandenen Bildpfade (all_paths).
        ?ober max_samples_per_epoch kann die effektive L-nge gedeckelt werden,
        damit der DataLoader weniger Triplets pro Epoche zieht,
        w-hrend __getitem__ unver-ndert augmentiert.
        """
        if self.total_available_samples == 0:
            return 0
        if not self.max_samples_per_epoch or self.max_samples_per_epoch <= 0:
            return self.total_available_samples
        return min(self.total_available_samples, self.max_samples_per_epoch)

    def _sample_negative_uuid(self, positive_uuid: str) -> str:
        # Schnelleres negatives Sampling
        idx = self.card_to_idx[positive_uuid]
        neg_idx = random.randint(0, len(self.card_ids) - 2)
        if neg_idx >= idx:
            neg_idx += 1
        return self.card_ids[neg_idx]

    def __getitem__(self, idx: int):
        pos_uuid = random.choice(self.card_ids)
        pos_list = self.card_to_paths[pos_uuid]
        a_path = random.choice(pos_list)
        p_path = random.choice(pos_list)
        # ensure p_path differs when possible
        if len(pos_list) > 1:
            while p_path == a_path:
                p_path = random.choice(pos_list)
        neg_uuid = self._sample_negative_uuid(pos_uuid)
        n_path = random.choice(self.card_to_paths[neg_uuid])

        # Nutze Bild-Caching (LRU) oder lade direkt
        if self.cache_images:
            a_img = self._image_cache.get(a_path)
            if a_img is None:
                a_img = Image.open(a_path).convert("RGB")
                self._image_cache[a_path] = a_img
            p_img = self._image_cache.get(p_path)
            if p_img is None:
                p_img = Image.open(p_path).convert("RGB")
                self._image_cache[p_path] = p_img
            n_img = self._image_cache.get(n_path)
            if n_img is None:
                n_img = Image.open(n_path).convert("RGB")
                self._image_cache[n_path] = n_img
        else:
            a_img = Image.open(a_path).convert("RGB")
            p_img = Image.open(p_path).convert("RGB")
            n_img = Image.open(n_path).convert("RGB")

        # OPTIONAL: Kamera-Augmentierung (wie bisher)
        if self.use_camera_augmentor:
            a_img = self.camera_augmentor.create_camera_like_augmentations(a_img, num_augmentations=1)[-1]
            p_img = self.camera_augmentor.create_camera_like_augmentations(p_img, num_augmentations=1)[-1]
            n_img = self.camera_augmentor.create_camera_like_augmentations(n_img, num_augmentations=1)[-1]

        # Effektive Pipeline: Scryfall-Bild -> (optional) CameraLikeAugmentor -> Set-Symbol-Crop -> Resize/ToTensor/Normalize.
        # NEU: Set-Symbol-Crops aus den augmentierten Bildern schneiden
        a_crop_img = crop_set_symbol(a_img, self.set_symbol_crop_cfg)
        p_crop_img = crop_set_symbol(p_img, self.set_symbol_crop_cfg)
        n_crop_img = crop_set_symbol(n_img, self.set_symbol_crop_cfg)

        # Vollbilder transformieren
        if self.transform_anchor:
            a_full_t = self.transform_anchor(a_img)
        else:
            a_full_t = T.ToTensor()(a_img)

        if self.transform_posneg:
            p_full_t = self.transform_posneg(p_img)
            n_full_t = self.transform_posneg(n_img)
        else:
            p_full_t = T.ToTensor()(p_img)
            n_full_t = T.ToTensor()(n_img)

        # NEU: Crops transformieren
        crop_transform = self.transform_crop or T.Compose([T.ToTensor()])
        a_crop_t = crop_transform(a_crop_img)
        p_crop_t = crop_transform(p_crop_img)
        n_crop_t = crop_transform(n_crop_img)

        label = self.card_to_idx[pos_uuid]

        # NEU: Vollbild + Crop pro Anchor/Pos/Neg zur-ckgeben
        return a_full_t, a_crop_t, p_full_t, p_crop_t, n_full_t, n_crop_t, label

    @property
    def num_classes(self) -> int:
        return len(self.card_ids)




