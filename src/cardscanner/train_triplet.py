import argparse
import json
import math
import os
import random
import sys
import time
from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torchvision.utils as vutils
from PIL import Image
import yaml
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.cardscanner.model import Encoder, save_encoder
from src.cardscanner.dataset import TripletImageDataset
from src.cardscanner.image_pipeline import (
    build_resize_normalize_transform,
    detect_image_size,
    get_set_symbol_crop_cfg,
)


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data:
        raise ValueError("Config file is empty")
    return data


def get_transforms(resize_hw: Tuple[int, int], variant: str = "full") -> Tuple[T.Compose, T.Compose]:
    """
    Liefert zwei Pipelines:
    - anchor_transform: darf staerker augmentieren (Rotation/Blur etc.).
    - base_transform: nur Resize + Norm, um CPU/RAM zu schonen.
    """
    base_transform = build_resize_normalize_transform(resize_hw)

    variant = (variant or "full").lower()
    if variant not in {"full", "light"}:
        print(f"[WARN] Unbekannte Transform-Variante '{variant}', fallback auf 'full'")
        variant = "full"

    if variant == "light":
        anchor_transform = T.Compose([
            T.Resize(resize_hw, antialias=True),
            T.RandomRotation(
                1.5,
                fill=0,
                interpolation=InterpolationMode.BICUBIC,
                expand=False,
            ),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        anchor_transform = T.Compose([
            T.Resize(resize_hw, antialias=True),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 0.8))], p=0.15),
            T.RandomRotation(
                2,
                fill=0,
                interpolation=InterpolationMode.BICUBIC,
                expand=False,
            ),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    return anchor_transform, base_transform


def save_feature_maps(feature_tensor: torch.Tensor, out_dir: str, max_channels: int = 16):
    """
    Speichert bis zu max_channels einzelne Feature-Maps unveraendert in Originalaufloesung.
    """
    os.makedirs(out_dir, exist_ok=True)
    fmap = feature_tensor[0]
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
    print(f"[DEBUG] Featuremaps gespeichert: {out_dir} ({c} Kanaele)")


def save_feature_maps_for_preview(model: nn.Module,
                                  preview_dir: str,
                                  featuremap_dir: str,
                                  resize_hw: Tuple[int, int],
                                  device: torch.device,
                                  preserve_symbol_crop: bool = True):
    full_dir = os.path.join(preview_dir, "full")
    symbol_dir = os.path.join(preview_dir, "symbol")
    if not os.path.isdir(full_dir):
        print("[DEBUG] Kein full/-Ordner im preview_dir -> Featuremaps werden uebersprungen.")
        return
    full_files = sorted([f for f in os.listdir(full_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not full_files:
        print("[DEBUG] Keine Preview-Bilder gefunden -> Featuremaps werden uebersprungen.")
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


def log_embeddings_for_projector(model: nn.Module,
                                 dataset: TripletImageDataset,
                                 writer: SummaryWriter,
                                 device: torch.device,
                                 epoch_idx: int,
                                 sample_size: int = 128):
    if sample_size <= 0 or len(dataset) == 0:
        return
    count = min(sample_size, len(dataset))
    samples: List[torch.Tensor] = []
    labels: List[int] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(count):
            anchor_full, _, _, _, _, _, label = dataset[random.randrange(len(dataset))]
            samples.append(anchor_full)
            labels.append(int(label))
        batch = torch.stack(samples).to(device)
        embeddings = model(batch)
        embeddings = F.normalize(embeddings, p=2, dim=-1).cpu()
    label_names = [dataset.card_ids[l] if l < len(dataset.card_ids) else str(l) for l in labels]
    writer.add_embedding(
        embeddings,
        metadata=label_names,
        tag="embeddings/full",
        global_step=epoch_idx,
    )
    if was_training:
        model.train()


def log_debug_images(writer: SummaryWriter,
                     tag_prefix: str,
                     tensors: torch.Tensor,
                     epoch: int):
    if tensors.numel() == 0:
        return
    grid = vutils.make_grid(tensors[:4].detach().cpu(), nrow=4, normalize=True, scale_each=True)
    writer.add_image(tag_prefix, grid, epoch)


def train(config: dict, images_dir: str, camera_dir: str | None = None) -> nn.Module | None:
    training_cfg = config["training"]
    model_cfg = config["model"]
    default_model_output = model_cfg.get("weights_path")
    hardware_cfg = config["hardware"]
    paths_cfg = config.get("paths", {})
    debug_cfg = config.get("debug", {})
    margin = training_cfg.get("margin", 0.3)
    ce_weight = training_cfg.get("ce_weight", 0.0)
    symbol_loss_weight = training_cfg.get("symbol_loss_weight", 0.5)

    fine_tune = bool(training_cfg.get("fine_tune", False))
    strict_load = bool(training_cfg.get("strict_load", fine_tune))
    load_model_path = training_cfg.get("model_path") or model_cfg.get("weights_path")
    save_model_path = training_cfg.get("save_path") or model_cfg.get("weights_path")

    print(f"[CONFIG] fine_tune={fine_tune}")
    print(f"[CONFIG] model_path={load_model_path}")
    print(f"[CONFIG] save_path={save_model_path}")
    print(f"[CONFIG] strict_load={strict_load}")

    if not save_model_path:
        raise SystemExit("[ERROR] Kein save_path/model.weights_path definiert – bitte config aktualisieren.")
    if fine_tune:
        if not load_model_path:
            raise SystemExit("[ERROR] Fine-Tuning verlangt training.model_path oder model.weights_path.")
        if not os.path.exists(load_model_path):
            raise SystemExit(f"[ERROR] Basis-Checkpoint nicht gefunden: {load_model_path}")
        if os.path.abspath(load_model_path) == os.path.abspath(save_model_path):
            raise SystemExit("[ERROR] save_path == model_path – Basismodell würde überschrieben. Bitte anderen Pfad wählen.")

    use_cuda = hardware_cfg.get("use_cuda", True) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Training on {device}")
    print(
        f"[INFO] Loss-Setup: margin={margin:.3f}, ce_weight={ce_weight:.3f}, symbol_loss_weight={symbol_loss_weight:.3f}"
    )

    if training_cfg.get("auto_detect_size", False):
        target_width, target_height = detect_image_size(images_dir)
    else:
        target_width = training_cfg["target_width"]
        target_height = training_cfg["target_height"]
        print(f"Using configured image size (width x height): {target_width}x{target_height}")
    resize_hw = (target_height, target_width)
    print(f"[INFO] Training resize (W x H): {target_width}x{target_height}")

    crop_cfg = get_set_symbol_crop_cfg(config) or {}
    crop_resize_hw = (
        crop_cfg.get("target_height", 64),
        crop_cfg.get("target_width", 160),
    )
    crop_transform = build_resize_normalize_transform(crop_resize_hw)

    transform_variant = training_cfg.get("transform_variant", "full")
    print(f"[INFO] Transform-Variante: {transform_variant}")
    aug_anchor, base = get_transforms(resize_hw, variant=transform_variant)

    cache_images = training_cfg.get("cache_images", True)
    cache_max_size = training_cfg.get("cache_max_size", 1000)

    dataset_camera_dir = None
    if training_cfg.get("use_camera_images"):
        dataset_camera_dir = training_cfg.get("camera_images_path") or camera_dir
        if dataset_camera_dir and os.path.isdir(dataset_camera_dir):
            print(f"[INFO] Zusätzliche Camera-Bilder aktiv: {dataset_camera_dir}")
        else:
            print("[WARN] use_camera_images=true, aber Pfad nicht gefunden – verwende nur Scryfall.")
            dataset_camera_dir = None
    elif camera_dir:
        print("[INFO] Kamera-Bilder werden ignoriert (nur Scryfall-Daten erlaubt).")

    max_samples = training_cfg.get("max_samples_per_epoch")
    if max_samples is not None and max_samples <= 0:
        max_samples = None
    if max_samples:
        print(f"[INFO] max_samples_per_epoch: {max_samples}")
    else:
        print("[INFO] max_samples_per_epoch: full dataset")
    use_camera_augmentor = training_cfg.get("use_camera_augmentor", True)
    if not use_camera_augmentor:
        print("[INFO] Kamera-Augmentor deaktiviert -> es laufen nur die Torch-Transforms.")

    dataset = TripletImageDataset(
        images_dir,
        dataset_camera_dir,
        aug_anchor,
        base,
        seed=training_cfg.get("seed", 42),
        use_camera_augmentor=use_camera_augmentor,
        augmentor_params=config.get("camera_augmentor", config.get("augmentation", {})),
        cache_images=cache_images,
        cache_max_size=cache_max_size,
        max_samples_per_epoch=max_samples,
        transform_crop=crop_transform,
        set_symbol_crop_cfg=crop_cfg,
    )

    print(f"\n[INFO] Trainingsdaten: {len(dataset.card_ids)} Karten (Scryfall)")
    print(f"[INFO] Verfuegbare Einzelbilder: {dataset.total_available_samples}")
    if max_samples:
        print(f"[INFO] Effektive Dataset-Laenge pro Epoche: {len(dataset)} (Limit {max_samples})")
    else:
        print(f"[INFO] Effektive Dataset-Laenge pro Epoche: {len(dataset)} (alle verfuegbaren Samples)")

    debug_preview_dir = debug_cfg.get("augmentation_preview_dir")
    featuremap_dir = debug_cfg.get("featuremap_preview_dir")
    if debug_preview_dir:
        dataset.save_augmentations_of_first_card(debug_preview_dir, n_aug=config["augmentation"].get("num_augmentations", 8))
    if featuremap_dir:
        preview_model = Encoder(
            embed_dim=model_cfg["embed_dim"],
            num_classes=len(dataset.card_ids),
            pretrained=True,
        ).to(device)
        save_feature_maps_for_preview(preview_model, debug_preview_dir or "", featuremap_dir, resize_hw, device)
        del preview_model

    if len(dataset) == 0:
        print("No training data found!")
        return None

    batch_size = training_cfg.get("batch_size", 24)
    num_workers = training_cfg.get("num_workers", training_cfg.get("workers", 2))
    if num_workers is not None and num_workers < 0:
        num_workers = 0
    prefetch_factor = training_cfg.get("prefetch_factor")
    if prefetch_factor is not None and prefetch_factor < 1:
        prefetch_factor = 1
    pin_memory = training_cfg.get("pin_memory", device.type == "cuda")

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )
    if num_workers and num_workers > 0 and prefetch_factor:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    print(f"[INFO] DataLoader: batch_size={batch_size}, num_workers={num_workers}, prefetch_factor={loader_kwargs.get('prefetch_factor', 'default')}")
    full_batches = len(loader)
    print(f"[INFO] Batches pro Epoche (ohne Limit): {full_batches}")
    max_batches = training_cfg.get("max_batches_per_epoch")
    if max_batches is not None and max_batches <= 0:
        max_batches = None
    if max_batches:
        print(f"[INFO] Limitiere Batches pro Epoche auf {max_batches}")
    effective_batches = min(full_batches, max_batches) if max_batches else full_batches

    model = Encoder(
        embed_dim=model_cfg["embed_dim"],
        num_classes=dataset.num_classes,
        pretrained=True,
    ).to(device)

    load_pretrained = training_cfg.get("load_pretrained_weights", True) or fine_tune
    if load_pretrained and load_model_path:
        if not os.path.exists(load_model_path):
            raise SystemExit(f"[ERROR] Gewichtsdatei nicht gefunden: {load_model_path}")
        print(f"[INFO] Lade Pretrained-Checkpoint: {load_model_path}")
        try:
            state = torch.load(load_model_path, map_location=device)
            checkpoint_meta = None
            if isinstance(state, dict) and "state_dict" in state:
                checkpoint_meta = state
                state = state["state_dict"]
            sample_key = next(iter(model.state_dict().keys()))
            before_sum = float(model.state_dict()[sample_key].sum().item())
            missing = model.load_state_dict(state, strict=strict_load)
            after_sum = float(model.state_dict()[sample_key].sum().item())
            if checkpoint_meta and "num_classes" in checkpoint_meta:
                print(f"[INFO] Checkpoint-Klassen (zur Kontrolle): {checkpoint_meta['num_classes']}")
            if missing.missing_keys:
                print(f"[WARN] Fehlende Keys beim Laden: {missing.missing_keys}")
            if math.isclose(before_sum, after_sum, rel_tol=1e-5, abs_tol=1e-5):
                print("[WARN] Checksum unverändert – bitte prüfen, ob der Checkpoint korrekt geladen wurde.")
        except Exception as exc:
            raise SystemExit(f"[ERROR] Konnte Gewichte nicht laden ({load_model_path}): {exc}") from exc
    elif load_pretrained and not load_model_path:
        print("[WARN] load_pretrained_weights=true, aber kein Pfad angegeben.")

    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg["learning_rate"])
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - nn.functional.cosine_similarity(x, y),
        margin=margin,
        reduction="mean",
    )
    ce_loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    global_step = 0
    batch_log_interval = training_cfg.get("tensorboard_batch_interval", 20)
    projector_interval = training_cfg.get("tensorboard_projector_interval", 5)
    projector_samples = training_cfg.get("tensorboard_projector_samples", 128)
    tb_verbose = debug_cfg.get("tensorboard_verbose", False)

    output_root = paths_cfg.get("output_dir", "./output")
    run_dir = os.path.join(output_root, "runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    print(f"[INFO] TensorBoard-Logs: {run_dir}")

    try:
        dummy_input = torch.randn(1, 3, resize_hw[0], resize_hw[1], device=device)
        writer.add_graph(model, dummy_input)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Konnte Modellgraph nicht loggen: {exc}")

    for epoch in range(training_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        running_triplet_full = 0.0
        running_triplet_crop = 0.0
        running_ce = 0.0
        running_sim_pos = 0.0
        running_sim_neg = 0.0
        processed_batches = 0
        total_samples = 0
        images_logged_this_epoch = False

        batch_iter = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{training_cfg['epochs']}",
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

            anchor_full_emb, anchor_logits = model(anchor_full, return_logits=True)
            pos_full_emb = model(positive_full)
            neg_full_emb = model(negative_full)
            anchor_crop_emb = model(anchor_crop)
            pos_crop_emb = model(positive_crop)
            neg_crop_emb = model(negative_crop)

            anchor_full_emb = nn.functional.normalize(anchor_full_emb, p=2, dim=-1)
            pos_full_emb = nn.functional.normalize(pos_full_emb, p=2, dim=-1)
            neg_full_emb = nn.functional.normalize(neg_full_emb, p=2, dim=-1)
            anchor_crop_emb = nn.functional.normalize(anchor_crop_emb, p=2, dim=-1)
            pos_crop_emb = nn.functional.normalize(pos_crop_emb, p=2, dim=-1)
            neg_crop_emb = nn.functional.normalize(neg_crop_emb, p=2, dim=-1)

            triplet_loss_full = triplet_loss_fn(anchor_full_emb, pos_full_emb, neg_full_emb)
            triplet_loss_crop = triplet_loss_fn(anchor_crop_emb, pos_crop_emb, neg_crop_emb)
            ce_loss = ce_loss_fn(anchor_logits, labels)

            loss = triplet_loss_full + symbol_loss_weight * triplet_loss_crop + ce_weight * ce_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                sim_pos = F.cosine_similarity(anchor_full_emb, pos_full_emb).mean().item()
                sim_neg = F.cosine_similarity(anchor_full_emb, neg_full_emb).mean().item()

            global_step += 1
            running_loss += loss.item()
            running_triplet_full += triplet_loss_full.item()
            running_triplet_crop += triplet_loss_crop.item()
            running_ce += ce_loss.item()
            running_sim_pos += sim_pos
            running_sim_neg += sim_neg
            processed_batches += 1
            total_samples += anchor_full.size(0)

            batch_iter.set_postfix({
                "loss": f"{loss.item():.4f}",
                "trip_full": f"{triplet_loss_full.item():.4f}",
                "trip_crop": f"{triplet_loss_crop.item():.4f}",
                "ce": f"{ce_loss.item():.4f}",
            })

            if batch_log_interval and (global_step % batch_log_interval == 0):
                writer.add_scalar("batch/loss_triplet_full", triplet_loss_full.item(), global_step)
                writer.add_scalar("batch/loss_triplet_crop", triplet_loss_crop.item(), global_step)
                writer.add_scalar("batch/loss_ce", ce_loss.item(), global_step)
                writer.add_scalar("batch/loss_total", loss.item(), global_step)
                writer.add_scalar("batch/similarity_positive", sim_pos, global_step)
                writer.add_scalar("batch/similarity_negative", sim_neg, global_step)

            if tb_verbose and (epoch % 2 == 0) and not images_logged_this_epoch:
                log_debug_images(writer, "samples/anchor_full", anchor_full, epoch)
                log_debug_images(writer, "samples/positive_full", positive_full, epoch)
                log_debug_images(writer, "samples/negative_full", negative_full, epoch)
                images_logged_this_epoch = True

            if max_batches and processed_batches >= max_batches:
                break

        avg_batches = max(1, processed_batches)
        avg_triplet_full = running_triplet_full / avg_batches
        avg_triplet_crop = running_triplet_crop / avg_batches
        avg_ce = running_ce / avg_batches
        avg_total = running_loss / avg_batches
        mean_sim_pos = running_sim_pos / avg_batches
        mean_sim_neg = running_sim_neg / avg_batches
        print(f"Epoch {epoch + 1}: loss={avg_total:.4f}")

        writer.add_scalar("epoch/loss_triplet_full", avg_triplet_full, epoch)
        writer.add_scalar("epoch/loss_triplet_crop", avg_triplet_crop, epoch)
        writer.add_scalar("epoch/loss_ce", avg_ce, epoch)
        writer.add_scalar("epoch/loss_total", avg_total, epoch)
        writer.add_scalar("epoch/similarity_positive_mean", mean_sim_pos, epoch)
        writer.add_scalar("epoch/similarity_negative_mean", mean_sim_neg, epoch)
        writer.add_scalar("epoch/similarity_gap", mean_sim_pos - mean_sim_neg, epoch)
        writer.add_scalar("epoch/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("epoch/train_samples", total_samples, epoch)
        if device.type == "cuda":
            writer.add_scalar("epoch/gpu_memory_mb", torch.cuda.memory_allocated(device) / (1024 ** 2), epoch)
        writer.add_text(
            "training/status",
            f"Epoch {epoch + 1}: total_loss={avg_total:.4f}, triplet_full={avg_triplet_full:.4f}, "
            f"triplet_crop={avg_triplet_crop:.4f}, ce={avg_ce:.4f}, sim_gap={(mean_sim_pos - mean_sim_neg):.3f}",
            epoch,
        )
        writer.flush()

        if projector_interval and (epoch + 1) % projector_interval == 0:
            log_embeddings_for_projector(model, dataset, writer, device, epoch + 1, projector_samples)

        if avg_total < best_loss:
            best_loss = avg_total
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if avg_total <= training_cfg["min_loss_threshold"]:
            print("Stopping because loss threshold reached")
            break
        if patience_counter >= training_cfg["early_stopping_patience"]:
            print("Stopping because patience limit reached")
            break

    writer.close()

    if best_state is None:
        print("Training did not produce a valid model")
        return None

    model.load_state_dict(best_state)
    model.eval()

    weights_path = save_model_path
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    save_encoder(model, weights_path, class_ids=dataset.card_ids)

    mapping_path = f"{weights_path}.classes.json"
    with open(mapping_path, "w", encoding="utf-8") as fh:
        json.dump({"class_ids": dataset.card_ids, "class_to_idx": dataset.card_to_idx}, fh, indent=2)
    print(f"Saved model to {weights_path}")
    print(f"Saved class mapping to {mapping_path}")
    if default_model_output and os.path.abspath(default_model_output) != os.path.abspath(weights_path):
        print(
            "[INFO] Hinweis: config.model.weights_path verweist aktuell auf "
            f"{default_model_output}. Bitte anpassen, damit Export/Erkennung das neue Modell nutzen."
        )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Triplet/CE model for MTG card embeddings.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    config = load_config(args.config)
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

