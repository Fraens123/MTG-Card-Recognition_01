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
from src.cardscanner.crop_utils import crop_set_symbol, load_symbol_crop_cfg
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
        precomputed_dir: Optional[str] = None,
        anchor_from_original: bool = True,
        slots_per_card: int = 1,
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
        self.precomputed_dir = precomputed_dir
        self.anchor_from_original = anchor_from_original
        self.slots_per_card = max(1, int(slots_per_card))

        # Set-Symbol-Konfiguration zentral laden
        self.set_symbol_crop_cfg = load_symbol_crop_cfg()
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
        self.card_to_precomputed: Dict[str, List[Tuple[str, Optional[str]]]] = {}
        self.samples: List[str] = []
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

        if self.precomputed_dir:
            self._load_precomputed_variants(self.precomputed_dir)

        self.sample_card_ids = [cid for cid in self.card_ids if self.card_to_paths.get(cid)]
        if not self.sample_card_ids:
            raise ValueError("TripletImageDataset: Keine gültigen Karten gefunden.")
        self.samples = [
            card_id
            for card_id in self.sample_card_ids
            for _ in range(self.slots_per_card)
        ]

    def _load_precomputed_variants(self, root_dir: str) -> None:
        if not os.path.isdir(root_dir):
            print(f"[INFO] Vorab-Augmentierungen: Verzeichnis nicht gefunden ({root_dir})")
            return
        total_variants = 0
        for current_root, dirs, _ in os.walk(root_dir):
            if "full" not in dirs:
                continue
            # Verhindere, dass os.walk in full/symbol weiter absteigt
            dirs[:] = [d for d in dirs if d not in ("full", "symbol")]
            card_dir = current_root
            card_uuid = os.path.basename(card_dir)
            full_dir = os.path.join(card_dir, "full")
            symbol_dir = os.path.join(card_dir, "symbol")
            variants: List[Tuple[str, Optional[str]]] = []
            for fname in sorted(os.listdir(full_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                full_path = os.path.join(full_dir, fname)
                symbol_path = None
                if os.path.isdir(symbol_dir):
                    candidate = os.path.join(symbol_dir, fname)
                    if os.path.exists(candidate):
                        symbol_path = candidate
                variants.append((full_path, symbol_path))
            if variants:
                self.card_to_precomputed[card_uuid] = variants
                total_variants += len(variants)
        print(
            f"[INFO] Vorab-Augmentierungen geladen: {len(self.card_to_precomputed)} Karten, "
            f"{total_variants} Varianten"
        )

    def _load_image(self, path: str) -> Image.Image:
        if self.cache_images and self._image_cache is not None:
            cached = self._image_cache.get(path)
            if cached is not None:
                return cached.copy()
        img = Image.open(path).convert("RGB")
        if self.cache_images and self._image_cache is not None:
            self._image_cache[path] = img
            return img.copy()
        return img

    def _load_precomputed_variant(self, card_id: str) -> Optional[Tuple[Image.Image, Image.Image]]:
        variants = self.card_to_precomputed.get(card_id)
        if not variants:
            return None
        full_path, symbol_path = random.choice(variants)
        full_img = self._load_image(full_path)
        if symbol_path:
            crop_img = self._load_image(symbol_path)
        else:
            crop_img = crop_set_symbol(full_img, self.set_symbol_crop_cfg)
        return full_img, crop_img

    def _generate_augmented_from_original(self, card_id: str) -> Tuple[Image.Image, Image.Image]:
        paths = self.card_to_paths.get(card_id)
        if not paths:
            raise ValueError(f"Keine Originalbilder für Karte {card_id} gefunden.")
        base_img = self._load_image(random.choice(paths))
        if self.use_camera_augmentor and hasattr(self, "camera_augmentor"):
            aug_img = self.camera_augmentor.create_camera_like_augmentations(
                base_img, num_augmentations=1
            )[-1]
        else:
            aug_img = base_img.copy()
        crop_img = crop_set_symbol(aug_img, self.set_symbol_crop_cfg)
        return aug_img, crop_img

    def _get_anchor_pair(self, card_id: str) -> Tuple[Image.Image, Image.Image]:
        if self.anchor_from_original and self.card_to_paths.get(card_id):
            anchor_img = self._load_image(random.choice(self.card_to_paths[card_id]))
            anchor_crop = crop_set_symbol(anchor_img, self.set_symbol_crop_cfg)
            return anchor_img, anchor_crop
        variant = self._load_precomputed_variant(card_id)
        if variant:
            return variant
        return self._generate_augmented_from_original(card_id)

    def _get_augmented_pair(self, card_id: str) -> Tuple[Image.Image, Image.Image]:
        variant = self._load_precomputed_variant(card_id)
        if variant:
            return variant
        return self._generate_augmented_from_original(card_id)
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
        return len(self.samples)

    def _sample_negative_uuid(self, positive_uuid: str) -> str:
        neg_uuid = random.choice(self.sample_card_ids)
        while neg_uuid == positive_uuid:
            neg_uuid = random.choice(self.sample_card_ids)
        return neg_uuid

    def __getitem__(self, idx: int):
        if not self.samples:
            raise IndexError("TripletImageDataset besitzt keine Samples.")

        pos_uuid = self.samples[idx % len(self.samples)]
        neg_uuid = self._sample_negative_uuid(pos_uuid)

        anchor_full_img, anchor_crop_img = self._get_anchor_pair(pos_uuid)
        positive_full_img, positive_crop_img = self._get_augmented_pair(pos_uuid)
        negative_full_img, negative_crop_img = self._get_augmented_pair(neg_uuid)

        if self.transform_anchor:
            a_full_t = self.transform_anchor(anchor_full_img)
        else:
            a_full_t = T.ToTensor()(anchor_full_img)

        if self.transform_posneg:
            p_full_t = self.transform_posneg(positive_full_img)
            n_full_t = self.transform_posneg(negative_full_img)
            a_crop_t = self.transform_posneg(anchor_crop_img)
            p_crop_t = self.transform_posneg(positive_crop_img)
            n_crop_t = self.transform_posneg(negative_crop_img)
        else:
            p_full_t = T.ToTensor()(positive_full_img)
            n_full_t = T.ToTensor()(negative_full_img)
            a_crop_t = T.ToTensor()(anchor_crop_img)
            p_crop_t = T.ToTensor()(positive_crop_img)
            n_crop_t = T.ToTensor()(negative_crop_img)

        label = self.card_to_idx[pos_uuid]
        return a_full_t, a_crop_t, p_full_t, p_crop_t, n_full_t, n_crop_t, label

    @property
    def num_classes(self) -> int:
        return len(self.card_ids)
