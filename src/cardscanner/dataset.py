from PIL import Image
from typing import Optional
# NEU: Set-Symbol-Crop-Funktion
def crop_set_symbol(img: Image.Image, crop_cfg: Optional[dict] = None) -> Image.Image:
    """
    Schneidet das Set-Symbol aus dem Bild.
    crop_cfg: dict mit Schlüsseln x_min, y_min, x_max, y_max (relativ 0..1).
    Wenn crop_cfg None ist, wird config.yaml geladen.
    """
    if crop_cfg is None:
        # Lazy load aus config.yaml
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            crop_cfg = config.get("debug", {}).get("set_symbol_crop", None)
        except Exception as e:
            print(f"[WARN] crop_set_symbol: kann config.yaml nicht laden: {e}")
            crop_cfg = None

    if not crop_cfg:
        # Fallback: gib Original zurück, damit nichts crasht
        return img

    x_min = float(crop_cfg.get("x_min", 0.7))
    y_min = float(crop_cfg.get("y_min", 0.3))
    x_max = float(crop_cfg.get("x_max", 0.95))
    y_max = float(crop_cfg.get("y_max", 0.6))

    w, h = img.size
    left   = int(x_min * w)
    top    = int(y_min * h)
    right  = int(x_max * w)
    bottom = int(y_max * h)

    # Grenzen clampen
    left   = max(0, min(left, w - 1))
    right  = max(left + 1, min(right, w))
    top    = max(0, min(top, h - 1))
    bottom = max(top + 1, min(bottom, h))

    return img.crop((left, top, right, bottom))

from collections import OrderedDict

# Picklebare LRUCache-Klasse für Bild-Caching
class LRUCache(OrderedDict):
    def __init__(self, max_size=1000):
        super().__init__()
        self.max_size = max_size
    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.max_size:
            self.popitem(last=False)

import yaml
from src.cardscanner.augment_cards import CameraLikeAugmentor
import os
import random
from typing import Dict, List, Tuple, Optional
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def parse_scryfall_filename(filename: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parst Scryfall-Dateinamen in verschiedenen Formaten:
    Format 1: {set_code}_{collector_number}_{card_name}_{card_uuid} (z.B. 10E_31_Pacifism_686352f6.jpg)
    Format 2: {set_code}-{collector_number}-{card_name} (z.B. war-65-rescuer-sphinx.png)
    Gibt zurück: (card_uuid, set_code, collector_number, card_name)
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

    # Robust: Wenn Format nicht passt, logge Warnung und gib None zurück
    print(f"[WARN] parse_scryfall_filename: Unbekanntes Format für '{filename}'")
    return None


class TripletImageDataset(Dataset):
    def __init__(
        self,
        scryfall_dir: str,
        camera_dir: Optional[str],
        transform_anchor: T.Compose = None,
        transform_posneg: T.Compose = None,
        seed: int = 42,
        use_camera_augmentor: bool = True,
        augmentor_params: dict = None,
        cache_images: bool = True,
        cache_max_size: int = 1000,
    ) -> None:
        """
        Dataset für MTG-Karten Triplet Training.
        Verwendet scryfall_dir als Quelle für augmentierte Bilder.
        """
        random.seed(seed)
        self.scryfall_dir = scryfall_dir  # Verzeichnis mit Originalbildern
        self.camera_dir = camera_dir
        self.transform_anchor = transform_anchor
        self.transform_posneg = transform_posneg
        self.use_camera_augmentor = use_camera_augmentor

        # NEU: Set-Symbol-Crop-Konfiguration aus config.yaml
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.set_symbol_crop_cfg = cfg.get("debug", {}).get("set_symbol_crop", None)
        except Exception as e:
            print(f"[WARN] TripletImageDataset: config.yaml nicht lesbar: {e}")
            self.set_symbol_crop_cfg = None
        if self.use_camera_augmentor:
            if augmentor_params is None:
                # Lade Parameter aus config.yaml
                with open("config.yaml", "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                augmentor_params = config.get("augmentation", {})
            self.camera_augmentor = CameraLikeAugmentor(
                brightness_range=(augmentor_params.get("brightness_min", 0.9), augmentor_params.get("brightness_max", 1.3)),
                contrast_range=(augmentor_params.get("contrast_min", 0.98), augmentor_params.get("contrast_max", 1.02)),
                blur_range=(0.0, augmentor_params.get("blur_max", 3.2)),
                noise_range=(0, augmentor_params.get("noise_max", 5.0)),
                rotation_range=(-augmentor_params.get("rotation_max", 5.0), augmentor_params.get("rotation_max", 5.0)),
                perspective=augmentor_params.get("perspective", 0.05),
                shadow=augmentor_params.get("shadow", 0.14),
                saturation_range=(augmentor_params.get("saturation_min", 0.8), augmentor_params.get("saturation_max", 1.2)),
                color_temperature_range=(augmentor_params.get("color_temperature_min", 0.84), augmentor_params.get("color_temperature_max", 1.16)),
                hue_shift_max=augmentor_params.get("hue_shift_max", 15.0),
                background_color=augmentor_params.get("background_color", "white"),
            )
        self.card_to_paths = {}
        if cache_images:
            self._image_cache = LRUCache(cache_max_size)
        else:
            self._image_cache = None
        self.cache_images = cache_images

        print(f"Lade Originalbilder aus: {self.scryfall_dir}")

        # Lade nur die Originalbilder (keine augmentierten Varianten)
        for root, _, files in os.walk(self.scryfall_dir):
            for f in files:
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                p = os.path.join(root, f)
                meta = parse_scryfall_filename(f)
                if not meta:
                    print(f"Kann nicht parsen: {f}")
                    continue
                card_uuid = meta[0]
                self.card_to_paths.setdefault(card_uuid, []).append(p)
                # Optionales Vorab-Caching (nur wenn cache_images und cache_max_size groß genug)
                if self.cache_images and len(self._image_cache) < self._image_cache.max_size:
                    try:
                        self._image_cache[p] = Image.open(p).convert("RGB")
                    except Exception as e:
                        print(f"[Cache] Fehler beim Laden von {p}: {e}")

        # Optional: Lade echte Kamera-Bilder
        if self.camera_dir and os.path.exists(self.camera_dir):
            print(f"Lade Camera-Bilder aus: {self.camera_dir}")
            for root, _, files in os.walk(self.camera_dir):
                for f in files:
                    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    p = os.path.join(root, f)
                    meta = parse_scryfall_filename(p)
                    if not meta:
                        continue
                    card_uuid = meta[0]
                    self.card_to_paths.setdefault(card_uuid, []).append(p)
                    # Optionales Vorab-Caching (nur wenn cache_images und cache_max_size groß genug)
                    if self.cache_images and len(self._image_cache) < self._image_cache.max_size:
                        try:
                            self._image_cache[p] = Image.open(p).convert("RGB")
                        except Exception as e:
                            print(f"[Cache] Fehler beim Laden von {p}: {e}")

        self.card_ids = sorted(list(self.card_to_paths.keys()))
        self.card_to_idx = {card_id: idx for idx, card_id in enumerate(self.card_ids)}
        self.all_paths = [p for paths in self.card_to_paths.values() for p in paths]

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
        Zusätzlich werden dazugehörige Set-Symbol-Crops gespeichert.
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
        from torchvision.utils import save_image

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
        print(f"Erzeuge {n_aug} Augmentierungen für Karte: {card_id} ({img_path})")

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
                save_image(aug_tensor, full_path)

                # Einfacher Symbol-Crop aus Original (nicht 100% identisch zur Transform-Pipeline)
                crop_img = crop_set_symbol(img, self.set_symbol_crop_cfg)
                crop_tensor = self.transform_anchor(crop_img) if self.transform_anchor else T.ToTensor()(crop_img)
                symbol_path = os.path.join(symbol_dir, f"{card_id}_aug_{i:02d}.png")
                save_image(crop_tensor, symbol_path)

        print(f"Augmentierungen gespeichert in: {output_dir}")

    def __len__(self) -> int:
        return max(1, len(self.all_paths))

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
        if self.transform_posneg:
            a_crop_t = self.transform_posneg(a_crop_img)
            p_crop_t = self.transform_posneg(p_crop_img)
            n_crop_t = self.transform_posneg(n_crop_img)
        else:
            a_crop_t = T.ToTensor()(a_crop_img)
            p_crop_t = T.ToTensor()(p_crop_img)
            n_crop_t = T.ToTensor()(n_crop_img)

        label = self.card_to_idx[pos_uuid]

        # NEU: Vollbild + Crop pro Anchor/Pos/Neg zurückgeben
        return a_full_t, a_crop_t, p_full_t, p_crop_t, n_full_t, n_crop_t, label

    @property
    def num_classes(self) -> int:
        return len(self.card_ids)
