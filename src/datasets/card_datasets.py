from __future__ import annotations

import os
import random
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.utils import save_image

from src.core.augmentations import CameraLikeAugmentor
from src.core.image_ops import crop_card_art, get_full_art_crop_cfg

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


class LRUCache(OrderedDict):
    """Einfacher LRU-Puffer f�r PIL-Images."""

    def __init__(self, max_size: int = 0):
        super().__init__()
        self.max_size = max(0, int(max_size))

    def get(self, key, default=None):
        if self.max_size <= 0:
            return default
        if key not in self:
            return default
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if self.max_size <= 0:
            return
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.max_size:
            self.popitem(last=False)


class AddGaussianNoise:
    """Torchvision-kompatibler Noise-Transform."""

    def __init__(self, std: float = 0.0):
        self.std = float(max(0.0, std))

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor


def _size_to_hw(size_cfg: Optional[Iterable[int]]) -> Tuple[int, int]:
    if not size_cfg:
        return 320, 224
    width, height = size_cfg
    return int(height), int(width)


def _build_card_transform(size_hw: Tuple[int, int], augment_cfg: Dict) -> T.Compose:
    transforms: List = [T.Resize(size_hw, antialias=True)]
    rotation = float(augment_cfg.get("rotation_deg", 0.0))
    if rotation > 0:
        transforms.append(T.RandomRotation(rotation, fill=0))

    brightness = augment_cfg.get("brightness")
    contrast = augment_cfg.get("contrast")
    color_kwargs = {}
    if brightness:
        color_kwargs["brightness"] = tuple(brightness)
    if contrast:
        color_kwargs["contrast"] = tuple(contrast)
    if color_kwargs:
        transforms.append(T.ColorJitter(**color_kwargs))

    blur_prob = float(augment_cfg.get("blur_prob", 0.0))
    if blur_prob > 0:
        transforms.append(T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=blur_prob))

    transforms.append(T.ToTensor())
    noise_std = float(augment_cfg.get("noise_std", 0.0))
    if noise_std > 0:
        transforms.append(AddGaussianNoise(noise_std))
    transforms.append(T.Normalize(DEFAULT_MEAN, DEFAULT_STD))
    return T.Compose(transforms)


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(DEFAULT_MEAN, device=tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(DEFAULT_STD, device=tensor.device).view(1, -1, 1, 1)
    tensor = tensor.detach().unsqueeze(0) if tensor.ndim == 3 else tensor.detach()
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def parse_scryfall_filename(filename: str) -> Optional[Tuple[str, str, str, str]]:
    """Parst verschiedene Scryfall-Dateiformate."""
    name, _ = os.path.splitext(os.path.basename(filename))
    parts = name.split("_")
    if len(parts) >= 4:
        set_code = parts[0]
        collector_number = parts[1]
        card_uuid = parts[-1]
        card_name = "_".join(parts[2:-1])
        return card_uuid, set_code, collector_number, card_name

    if "-" in name:
        parts = name.split("-", 2)
        if len(parts) == 3:
            set_code, collector_number, card_name = parts
            card_uuid = f"{set_code}_{collector_number}_{abs(hash(name)) % 10_000_000:07d}"
            return card_uuid, set_code, collector_number, card_name
    return None


def _index_card_images(directory: Optional[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    if not directory or not os.path.isdir(directory):
        return mapping
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        meta = parse_scryfall_filename(name)
        if not meta:
            continue
        card_uuid = meta[0]
        mapping.setdefault(card_uuid, []).append(path)
    return mapping


def _merge_card_mappings(base: Dict[str, List[str]], extra: Dict[str, List[str]]) -> None:
    for key, paths in extra.items():
        base.setdefault(key, []).extend(paths)


class CoarseDataset(Dataset):
    """
    Liefert Full-Card-Bild + Klassen-Label fuer das CE-Vortraining.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.images_cfg = cfg.get("images", {})
        coarse_cfg = cfg.get("training", {}).get("coarse", {})
        augment_cfg = coarse_cfg.get("augment", {})

        self.full_crop_cfg = get_full_art_crop_cfg(cfg)
        self.full_transform = _build_card_transform(_size_to_hw(self.images_cfg.get("full_card_size")), augment_cfg)

        self.card_to_paths = _index_card_images(self.paths.get("scryfall_dir"))
        if not self.card_to_paths:
            raise RuntimeError("CoarseDataset: keine Scryfall-Bilder gefunden.")
        self.card_ids = sorted(self.card_to_paths.keys())
        self.card_to_idx = {card_id: idx for idx, card_id in enumerate(self.card_ids)}
        self.samples = [(card_id, path) for card_id, paths in self.card_to_paths.items() for path in paths]
        cache_size = 0
        if coarse_cfg.get("cache_images", False):
            cache_size = int(coarse_cfg.get("cache_size", 0))
        # Hält dekodierte PIL-Images im RAM, um wiederholte Disk-Reads zu vermeiden.
        self.image_cache = LRUCache(max_size=cache_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        card_id, img_path = self.samples[index % len(self.samples)]
        img = self._load_image(img_path)
        full_img = crop_card_art(img, self.full_crop_cfg)

        full_tensor = self.full_transform(full_img)
        label = torch.tensor(self.card_to_idx[card_id], dtype=torch.long)
        return full_tensor, label

    def _load_image(self, path: str) -> Image.Image:
        cached = self.image_cache.get(path)
        if cached is not None:
            return cached.copy()
        # Disk-Read nur bei Cache-Miss (I/O-Flaschenhals vermeiden).
        img = Image.open(path).convert("RGB")
        self.image_cache[path] = img.copy()
        return img

    def save_augmentations_of_first_card(self, output_dir: str, n_aug: int = 8) -> None:
        if not self.samples:
            return
        os.makedirs(output_dir, exist_ok=True)
        card_id, img_path = self.samples[0]
        img = self._load_image(img_path)
        for i in range(n_aug):
            full_tensor = self.full_transform(crop_card_art(img, self.full_crop_cfg))
            save_image(_denormalize(full_tensor), os.path.join(output_dir, f"{card_id}_full_{i:02d}.png"))


class TripletImageDataset(Dataset):
    """
    Triplet-Dataset fuer das Fine-Tuning mit schwerer Kamera-Augmentierung.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.images_cfg = cfg.get("images", {})
        fine_cfg = cfg.get("training", {}).get("fine", {})
        augment_cfg = fine_cfg.get("augment", {})

        self.full_crop_cfg = get_full_art_crop_cfg(cfg)

        self.full_transform = _build_card_transform(_size_to_hw(self.images_cfg.get("full_card_size")), augment_cfg)

        self.card_to_paths = _index_card_images(self.paths.get("scryfall_dir"))
        if not self.card_to_paths:
            raise RuntimeError("TripletDataset: keine Trainingsbilder in scryfall_dir gefunden.")

        camera_dir = self.paths.get("camera_dir")
        self.use_camera_images = bool(fine_cfg.get("use_camera_images", True))
        if self.use_camera_images and camera_dir and os.path.isdir(camera_dir):
            _merge_card_mappings(self.card_to_paths, _index_card_images(camera_dir))
        elif not self.use_camera_images:
            print("[INFO] TripletDataset: Kamera-Bilder werden explizit nicht verwendet (use_camera_images=false).")

        self.card_ids = sorted(self.card_to_paths.keys())
        self.card_to_idx = {card_id: idx for idx, card_id in enumerate(self.card_ids)}
        self.total_images = sum(len(p) for p in self.card_to_paths.values())
        self.max_triplets = self.total_images
        cache_size = 0
        if fine_cfg.get("cache_images", False):
            cache_size = int(fine_cfg.get("cache_size", 0))
        # Triplet-Loader profitiert stark von gecachten PIL-Images (weniger I/O-Latenz).
        self.image_cache = LRUCache(max_size=cache_size)

        self.camera_augmentor: Optional[CameraLikeAugmentor] = None
        self.camera_aug_repeats = max(1, int(round(augment_cfg.get("camera_like_strength", 1))))
        if augment_cfg.get("camera_like"):
            self.camera_augmentor = CameraLikeAugmentor(**_map_camera_aug_params(augment_cfg))

    def __len__(self) -> int:
        return max(self.max_triplets, 1)

    def __getitem__(self, index: int):
        anchor_uuid = random.choice(self.card_ids)
        positive_paths = self.card_to_paths[anchor_uuid]
        anchor_path = random.choice(positive_paths)
        positive_path = random.choice(positive_paths)
        if len(positive_paths) > 1:
            while positive_path == anchor_path:
                positive_path = random.choice(positive_paths)
        negative_uuid = self._sample_negative_uuid(anchor_uuid)
        negative_path = random.choice(self.card_to_paths[negative_uuid])

        anchor_img = self._prepare_image(anchor_path)
        positive_img = self._prepare_image(positive_path)
        negative_img = self._prepare_image(negative_path)

        a_full = self.full_transform(crop_card_art(anchor_img, self.full_crop_cfg))
        p_full = self.full_transform(crop_card_art(positive_img, self.full_crop_cfg))
        n_full = self.full_transform(crop_card_art(negative_img, self.full_crop_cfg))
        label = torch.tensor(self.card_to_idx[anchor_uuid], dtype=torch.long)

        return a_full, p_full, n_full, label

    def _prepare_image(self, path: str) -> Image.Image:
        img = self._load_image(path)
        if self.camera_augmentor is not None:
            aug_images = self.camera_augmentor.create_camera_like_augmentations(img, num_augmentations=self.camera_aug_repeats)
            img = aug_images[-1]
        return img

    def _load_image(self, path: str) -> Image.Image:
        cached = self.image_cache.get(path)
        if cached is not None:
            return cached.copy()
        # Disk-Read nur bei Cache-Miss (I/O-Flaschenhals vermeiden).
        img = Image.open(path).convert("RGB")
        self.image_cache[path] = img.copy()
        return img

    def _sample_negative_uuid(self, positive_uuid: str) -> str:
        neg_uuid = random.choice(self.card_ids)
        while neg_uuid == positive_uuid:
            neg_uuid = random.choice(self.card_ids)
        return neg_uuid

    def save_augmentations_of_first_card(self, output_dir: str, n_aug: int = 12) -> None:
        if not self.card_ids:
            return
        os.makedirs(output_dir, exist_ok=True)
        card_id = self.card_ids[0]
        img_path = self.card_to_paths[card_id][0]
        for idx in range(n_aug):
            img = self._prepare_image(img_path)
            full_tensor = self.full_transform(crop_card_art(img, self.full_crop_cfg))
            save_image(_denormalize(full_tensor), os.path.join(output_dir, f"{card_id}_triplet_full_{idx:02d}.png"))


def _map_camera_aug_params(augment_cfg: Dict) -> Dict:
    params: Dict = {}
    if "brightness" in augment_cfg:
        params["brightness_range"] = tuple(augment_cfg["brightness"])
    if "contrast" in augment_cfg:
        params["contrast_range"] = tuple(augment_cfg["contrast"])
    if "noise_std" in augment_cfg:
        params["noise_std_max"] = float(augment_cfg["noise_std"])
    if "rotation_deg" in augment_cfg:
        params["tilt_deg_max"] = float(augment_cfg["rotation_deg"])
    if "blur_prob" in augment_cfg:
        params["blur_prob"] = float(augment_cfg["blur_prob"])
    params.setdefault("noise_prob", 0.6)
    params.setdefault("rotation_prob", 0.6)
    return params
