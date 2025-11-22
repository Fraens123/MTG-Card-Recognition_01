import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_utils import get_training_config, load_config
from src.datasets.card_datasets import TripletImageDataset
from src.core.model_builder import load_encoder, save_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-Training mit Triplet-Loss.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Pfad zur Konfigurationsdatei")
    return parser.parse_args()


def _build_dataloader(dataset: TripletImageDataset, train_cfg: dict) -> DataLoader:
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", min(8, os.cpu_count() or 2)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=4,
    )


def _freeze_backbone_layers(backbone: nn.Module, freeze_ratio: float) -> None:
    layers = list(backbone.children())
    if not layers:
        for param in backbone.parameters():
            param.requires_grad = False
        return
    num_freeze = max(1, int(len(layers) * freeze_ratio))
    for module in layers[:num_freeze]:
        for param in module.parameters():
            param.requires_grad = False


def _freeze_model(model: nn.Module, freeze_ratio: float) -> None:
    freeze_ratio = max(0.0, min(1.0, freeze_ratio))
    if freeze_ratio == 0:
        return
    _freeze_backbone_layers(model.backbone, freeze_ratio)


def _compute_losses(
    logits: torch.Tensor, labels: torch.Tensor, criterion: nn.Module
) -> torch.Tensor:
    return criterion(logits, labels)


def _build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    sched_cfg = train_cfg.get("scheduler") or {}
    sched_type = str(sched_cfg.get("type", "")).lower()
    if sched_type != "warmup_cosine":
        return None

    total_epochs = int(train_cfg.get("epochs", 1))
    warmup_epochs = max(0, int(sched_cfg.get("warmup_epochs", 1)))
    if warmup_epochs >= total_epochs:
        warmup_epochs = max(total_epochs - 1, 0)
    cosine_epochs = max(total_epochs - warmup_epochs, 1)

    start_factor = float(sched_cfg.get("warmup_start_factor", 0.1))
    eta_min = float(sched_cfg.get("min_lr", 0.0))

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        total_iters=max(warmup_epochs, 1),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=eta_min,
    )
    if warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine

    base_lr = optimizer.param_groups[0]["lr"]
    print(f"[SCHED] warmup_cosine: base_lr={base_lr} warmup_epochs={warmup_epochs} eta_min={eta_min}")
    return scheduler


def main() -> None:
    start_time = time.time()
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = get_training_config(cfg, "fine")
    batch_size = int(train_cfg.get("batch_size", 32))
    torch.backends.cudnn.benchmark = True  # schnelleres Convolution-Tuning fuer stabile Input-Shapes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Fine-Training auf {device}")

    dataset = TripletImageDataset(cfg)
    use_camera = getattr(dataset, "use_camera_images", True)
    print(f"[DATA] Karten={len(dataset.card_ids)} | Samples≈{len(dataset)} | Kamera-Bilder={'aktiv' if use_camera else 'deaktiviert'}")
    preview_dir = cfg.get("debug", {}).get("augmentation_preview_dir")
    if preview_dir:
        print(f"[DEBUG] Schreibe Triplet-Augmentierungen nach {preview_dir}")
        dataset.save_augmentations_of_first_card(preview_dir)
    dataloader = _build_dataloader(dataset, train_cfg)
    print(f"[LOAD] batch_size={batch_size} | Schritte pro Epoche={len(dataloader)}")

    coarse_path = os.path.join(cfg["paths"]["models_dir"], "encoder_coarse.pt")
    if not os.path.exists(coarse_path):
        raise FileNotFoundError(f"Coarse-Checkpoint nicht gefunden: {coarse_path}")

    model = load_encoder(coarse_path, cfg, num_classes=len(dataset.card_ids), device=device)
    model.num_classes = len(dataset.card_ids)
    print(f"[MODEL] Init von {coarse_path} | Klassen={model.num_classes}")

    freeze_ratio = float(train_cfg.get("freeze_ratio", 0.6))
    _freeze_model(model, freeze_ratio)

    lr = float(train_cfg.get("lr", 1e-4))
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = _build_scheduler(optimizer, train_cfg)
    margin = float(train_cfg.get("margin", 0.5))
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    ce_loss = nn.CrossEntropyLoss()
    triplet_weight = float(train_cfg.get("triplet_weight", 1.0))
    ce_weight = float(train_cfg.get("ce_weight", 0.2))

    hard_neg_enabled = bool(train_cfg.get("hard_negatives", {}).get("enabled", False))
    print(f"[TRAIN] Run 3A Fine-Tuning ohne HardNeg: freeze_ratio={freeze_ratio}, margin={margin}, lr={lr}")

    debug_root = cfg.get("paths", {}).get("debug_dir", "./debug")
    log_dir = os.path.join(debug_root, "logs", "fine")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[LOG] TensorBoard unter {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_scalar("hp/freeze_ratio", freeze_ratio, 0)
    writer.add_scalar("hp/margin", margin, 0)
    writer.add_scalar("hp/lr", lr, 0)
    writer.add_scalar("hp/batch_size", batch_size, 0)
    writer.add_scalar("hp/hard_neg_enabled", int(hard_neg_enabled), 0)

    # Hyperparameter kurz loggen
    sched_cfg = train_cfg.get("scheduler", {})
    print(
        "[HP] fine: "
        f"epochs={train_cfg.get('epochs')} "
        f"bs={train_cfg.get('batch_size')} "
        f"lr={lr} "
        f"num_workers={train_cfg.get('num_workers')} "
        f"cache_images={train_cfg.get('cache_images')} "
        f"cache_size={train_cfg.get('cache_size')} "
        f"sched={sched_cfg.get('type') or 'none'} "
        f"triplet_w={triplet_weight} ce_w={ce_weight} "
        f"margin={margin} freeze_ratio={freeze_ratio}"
    )

    best_loss = float("inf")
    best_state = None

    epochs = int(train_cfg.get("epochs", 1))
    if scheduler is not None:
        scheduler.step()
    for epoch in range(epochs):
        model.train()
        running_triplet = 0.0
        running_ce = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            anchor_full, pos_full, neg_full, labels = [b.to(device, non_blocking=True) for b in batch]
            optimizer.zero_grad()

            anchor_emb, anchor_logits = model(anchor_full, return_logits=True)
            if anchor_logits is None:
                raise RuntimeError("Classifier ist nicht konfiguriert (num_classes fehlt).")
            pos_emb = model(pos_full)
            neg_emb = model(neg_full)

            loss_triplet = triplet_loss(anchor_emb, pos_emb, neg_emb)
            loss_ce_total = _compute_losses(anchor_logits, labels, ce_loss)
            total_loss = triplet_weight * loss_triplet + ce_weight * loss_ce_total
            total_loss.backward()
            optimizer.step()

            running_triplet += loss_triplet.item()
            running_ce += loss_ce_total.item()

        avg_triplet = running_triplet / len(dataloader)
        avg_ce = running_ce / len(dataloader)
        total = triplet_weight * avg_triplet + ce_weight * avg_ce

        writer.add_scalar("loss/triplet", avg_triplet, epoch)
        writer.add_scalar("loss/ce", avg_ce, epoch)
        writer.add_scalar("loss/total", total, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)
        print(
            f"[Epoch {epoch + 1}] total={total:.4f} triplet={avg_triplet:.4f} CE={avg_ce:.4f} "
            f"w_triplet={triplet_weight} w_ce={ce_weight} lr={current_lr:.6f}"
        )
        if scheduler is not None:
            scheduler.step()
        if total < best_loss:
            best_loss = total
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    writer.close()
    if best_state is None:
        raise RuntimeError("Fine-Training fehlgeschlagen: kein best_state gefunden.")

    model.load_state_dict(best_state, strict=False)
    model.eval()

    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    out_path = os.path.join(cfg["paths"]["models_dir"], "encoder_fine.pt")
    save_encoder(model, out_path, card_ids=dataset.card_ids)
    print(f"[INFO] Feingetrimmtes Modell gespeichert unter {out_path}")
    elapsed = time.time() - start_time
    print(f"[TIME] fine-training abgeschlossen in {elapsed/60:.2f} min")
    print("[NEXT] Embeddings exportieren: python -m src.training.export_embeddings --config config.yaml")


if __name__ == "__main__":
    main()


