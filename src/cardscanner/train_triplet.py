import matplotlib.pyplot as plt
import math
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import yaml
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.cardscanner.model import Encoder, save_encoder
from src.cardscanner.dataset import TripletImageDataset


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data:
        raise ValueError("Config file is empty")
    return data


def detect_image_size(images_dir: str) -> Tuple[int, int]:
    for root, _, files in os.walk(images_dir):
        for name in files:
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(root, name)
            with Image.open(img_path) as img:
                width, height = img.size
            max_dim = 400
            if max(width, height) > max_dim:
                scale = max_dim / max(width, height)
                width = int(width * scale)
                height = int(height * scale)
            width = (width // 8) * 8 or 224
            height = (height // 8) * 8 or 320
            print(f"Detected training size: {width}x{height}")
            return width, height
    print("Falling back to 224x320")
    return 224, 320


def get_transforms(resize_hw: Tuple[int, int]) -> Tuple[T.Compose, T.Compose]:
    """
    Liefert Trainings- und Validierungs-Transforms für gegebene Zielgröße (H, W).
    """
    train_transform = T.Compose([
        T.Resize(resize_hw, antialias=True),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize(resize_hw, antialias=True),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform
def save_feature_maps(feature_tensor: torch.Tensor, out_dir: str, max_channels: int = 16):
    """
    feature_tensor: [1, C, H, W]
    Speichert bis zu max_channels einzelne Feature-Maps unverändert in Originalauflösung.
    """
    os.makedirs(out_dir, exist_ok=True)
    fmap = feature_tensor[0]  # [C, H, W]
    c = min(max_channels, fmap.shape[0])
    for i in range(c):
        ch = fmap[i].detach().cpu()
        ch_min = ch.min()
        ch_max = ch.max()
        if (ch_max - ch_min) > 1e-6:
            ch_norm = (ch - ch_min) / (ch_max - ch_min)
        else:
            ch_norm = torch.zeros_like(ch)
        nd = (ch_norm.numpy() * 255).astype("uint8")
        img = Image.fromarray(nd, mode="L")
        img.save(os.path.join(out_dir, f"ch_{i:03d}.png"))
    print(f"[DEBUG] Featuremaps gespeichert: {out_dir} ({c} Kanäle)")

def save_feature_maps_for_preview(model: nn.Module,
                                  preview_dir: str,
                                  featuremap_dir: str,
                                  resize_hw: Tuple[int, int],
                                  device: torch.device,
                                  preserve_symbol_crop: bool = True):
    full_dir = os.path.join(preview_dir, "full")
    symbol_dir = os.path.join(preview_dir, "symbol")
    if not os.path.isdir(full_dir):
        print("[DEBUG] Kein full/-Ordner im preview_dir – Featuremaps werden übersprungen.")
        return
    full_files = sorted([f for f in os.listdir(full_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not full_files:
        print("[DEBUG] Keine Preview-Bilder gefunden – Featuremaps werden übersprungen.")
        return
    full_path = os.path.join(full_dir, full_files[0])
    symbol_path = None
    if os.path.isdir(symbol_dir):
        symbol_files = sorted([f for f in os.listdir(symbol_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if symbol_files:
            symbol_path = os.path.join(symbol_dir, symbol_files[0])
    full_transform = T.Compose([
        T.Resize(resize_hw, antialias=True),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    symbol_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    def run_and_save(img_path: str, kind: str):
        img = Image.open(img_path).convert("RGB")
        if kind == "symbol" and preserve_symbol_crop:
            tensor = symbol_transform(img)
        else:
            tensor = full_transform(img)
        x = tensor.unsqueeze(0).to(device)
        feature_maps = {}
        def make_hook(name):
            def hook(module, inp, out):
                feature_maps[name] = out.detach().cpu()
            return hook
        layers = {
            "conv1": model.backbone[0],
            "layer1": model.backbone[4],
            "layer4": model.backbone[7],
        }
        handles = []
        for name, layer in layers.items():
            handles.append(layer.register_forward_hook(make_hook(name)))
        with torch.no_grad():
            _ = model(x)
        for h in handles:
            h.remove()
        for name, fmap in feature_maps.items():
            out_dir = os.path.join(featuremap_dir, kind, name)
            save_feature_maps(fmap, out_dir)
    run_and_save(full_path, "full")
    if symbol_path:
        run_and_save(symbol_path, "symbol")
    aug_anchor = T.Compose([
        T.Resize(resize_hw, antialias=True),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        T.RandomRotation(2, fill=0),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 0.8))], p=0.15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    base = T.Compose([
        T.Resize(resize_hw, antialias=True),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return aug_anchor, base


def train(config: dict, images_dir: str, camera_dir: str | None = None) -> nn.Module | None:
    training_cfg = config["training"]
    model_cfg = config["model"]
    hardware_cfg = config["hardware"]

    use_cuda = hardware_cfg.get("use_cuda", True) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Training on {device}")

    if training_cfg.get("auto_detect_size", False):
        target_width, target_height = detect_image_size(images_dir)
    else:
        target_width = training_cfg["target_width"]
        target_height = training_cfg["target_height"]
        print(f"Using configured image size (width x height): {target_width}x{target_height}")

    resize_hw = (target_height, target_width)

    aug_anchor, base = get_transforms(resize_hw)
    # Caching-Parameter aus config.yaml holen
    cache_images = config["training"].get("cache_images", True)
    cache_max_size = config["training"].get("cache_max_size", 1000)
    use_camera_images = training_cfg.get("use_camera_images", False)
    dataset_camera_dir = camera_dir if (use_camera_images and camera_dir and os.path.exists(camera_dir)) else None
    if camera_dir and not use_camera_images:
        print("[INFO] Kamera-Bilder werden für das Training deaktiviert (training.use_camera_images = False).")

    dataset = TripletImageDataset(
        images_dir,
        dataset_camera_dir,
        aug_anchor,
        base,
        seed=training_cfg.get("seed", 42),
        use_camera_augmentor=True,
        augmentor_params=config.get("augmentation", {}),
        cache_images=cache_images,
        cache_max_size=cache_max_size,
    )
    print(f"\n[INFO] Trainingsdaten: {len(dataset.card_ids)} Karten (Originalbilder)")
    print(f"[INFO] Augmentierungen pro Karte: {config['augmentation'].get('num_augmentations', 1)} (on-the-fly im RAM)")
    print(f"[INFO] Erwartete Embeddings pro Karte: {config['augmentation'].get('num_augmentations', 1)}")
    print(f"[INFO] Gesamtzahl Trainingsdaten (theoretisch): {len(dataset.card_ids) * config['augmentation'].get('num_augmentations', 1)}")

    # Initialisiere Modell vor Preview
    model = Encoder(
        embed_dim=model_cfg["embed_dim"],
        num_classes=len(dataset.card_ids),
        pretrained=True,
    ).to(device)

    # Augmentierungen der ersten Karte abspeichern
    n_aug = config["augmentation"].get("num_augmentations", 20)
    preview_dir = config.get("debug", {}).get("augmentation_preview_dir", "./debug/augment_preview")
    dataset.save_augmentations_of_first_card(preview_dir, n_aug=n_aug)
    featuremap_dir = config.get("debug", {}).get("featuremap_preview_dir", "./debug/feature_maps")
    save_feature_maps_for_preview(model, preview_dir, featuremap_dir, resize_hw, device)
    if len(dataset) == 0:
        print("No training data found!")
        return None

    loader = DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg["workers"],
        drop_last=True,
    )

    print(f"Classes (card UUIDs): {dataset.num_classes}")
    print(f"[INFO] Trainingsdateien (inkl. Kamera): {len(dataset.all_paths)}")
    full_batches = len(loader)
    print(f"[INFO] Batches pro Epoche (ohne Limit): {full_batches}")
    if training_cfg.get("max_batches_per_epoch"):
        print(f"[INFO] Limitiere Batches pro Epoche auf {training_cfg['max_batches_per_epoch']}")

    max_batches = training_cfg.get("max_batches_per_epoch")
    effective_batches = min(full_batches, max_batches) if max_batches else full_batches

    model = Encoder(
        embed_dim=model_cfg["embed_dim"],
        num_classes=dataset.num_classes,
        pretrained=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg["learning_rate"])
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - nn.functional.cosine_similarity(x, y),
        margin=training_cfg["margin"],
        reduction="mean",
    )
    ce_loss_fn = nn.CrossEntropyLoss()
    ce_weight = training_cfg.get("ce_weight", 0.0)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    symbol_loss_weight = training_cfg.get("symbol_loss_weight", 0.5)

    for epoch in range(training_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        processed_batches = 0
        batch_iter = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{training_cfg['epochs']}",
            unit="batch",
            total=effective_batches,
        )
        for batch in batch_iter:
            (
                anchor_full,
                anchor_crop,
                positive_full,
                positive_crop,
                negative_full,
                negative_crop,
                labels,
            ) = batch

            anchor_full = anchor_full.to(device, non_blocking=True)
            anchor_crop = anchor_crop.to(device, non_blocking=True)
            positive_full = positive_full.to(device, non_blocking=True)
            positive_crop = positive_crop.to(device, non_blocking=True)
            negative_full = negative_full.to(device, non_blocking=True)
            negative_crop = negative_crop.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Vollbild-Embeddings (mit CE-Logits)
            anchor_full_emb, anchor_logits = model(anchor_full, return_logits=True)
            pos_full_emb = model(positive_full)
            neg_full_emb = model(negative_full)

            # Crop-Embeddings (gleicher Encoder)
            anchor_crop_emb = model(anchor_crop)
            pos_crop_emb = model(positive_crop)
            neg_crop_emb = model(negative_crop)

            # Normierung
            anchor_full_emb = nn.functional.normalize(anchor_full_emb, p=2, dim=-1)
            pos_full_emb = nn.functional.normalize(pos_full_emb, p=2, dim=-1)
            neg_full_emb = nn.functional.normalize(neg_full_emb, p=2, dim=-1)

            anchor_crop_emb = nn.functional.normalize(anchor_crop_emb, p=2, dim=-1)
            pos_crop_emb = nn.functional.normalize(pos_crop_emb, p=2, dim=-1)
            neg_crop_emb = nn.functional.normalize(neg_crop_emb, p=2, dim=-1)

            # Triplet-Loss Vollbild
            triplet_loss_full = triplet_loss_fn(anchor_full_emb, pos_full_emb, neg_full_emb)
            # Triplet-Loss Symbol-Crop
            triplet_loss_crop = triplet_loss_fn(anchor_crop_emb, pos_crop_emb, neg_crop_emb)

            # CE-Loss nur auf Vollbild
            ce_loss = ce_loss_fn(anchor_logits, labels)

            loss = triplet_loss_full + symbol_loss_weight * triplet_loss_crop + ce_weight * ce_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            processed_batches += 1
            batch_iter.set_postfix({
                "loss": f"{loss.item():.4f}",
                "trip_full": f"{triplet_loss_full.item():.4f}",
                "trip_crop": f"{triplet_loss_crop.item():.4f}",
                "ce": f"{ce_loss.item():.4f}",
            })
            if max_batches and processed_batches >= max_batches:
                break

        avg_loss = running_loss / max(1, processed_batches)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if avg_loss <= training_cfg["min_loss_threshold"]:
            print("Stopping because loss threshold reached")
            break
        if patience_counter >= training_cfg["early_stopping_patience"]:
            print("Stopping because patience limit reached")
            break

    if best_state is None:
        print("Training did not produce a valid model")
        return None

    model.load_state_dict(best_state)
    model.eval()

    weights_path = model_cfg["weights_path"]
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    save_encoder(model, weights_path, class_ids=dataset.card_ids)

    mapping_path = f"{weights_path}.classes.json"
    with open(mapping_path, "w", encoding="utf-8") as fh:
        json.dump({"class_ids": dataset.card_ids, "class_to_idx": dataset.card_to_idx}, fh, indent=2)
    print(f"Saved model to {weights_path}")
    print(f"Saved class mapping to {mapping_path}")
    return model


def main() -> None:
    start = time.time()
    config = load_config("config.yaml")
    data_cfg = config["data"]
    model = train(config, data_cfg["scryfall_images"], data_cfg.get("camera_images"))
    end = time.time()
    if model is None:
        print("Training failed")
    else:
        print("Training finished successfully")
    print(f"Total runtime: {end - start:.2f}s")


if __name__ == "__main__":
    main()
