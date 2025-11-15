import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

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
from src.datasets.card_datasets import CoarseDataset, DatasetSubset
from src.core.model_builder import build_card_encoder, save_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coarse-Training des Karten-Encoders.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Pfad zur Konfigurationsdatei")
    return parser.parse_args()


def _compute_losses(
    logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    ce_full_weight: float,
    ce_symbol_weight: float,
    ce_name_weight: float,
    criterion: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_full, logits_symbol, logits_name = logits
    device = labels.device
    loss_full = criterion(logits_full, labels) if logits_full is not None else torch.zeros(1, device=device)
    loss_symbol = criterion(logits_symbol, labels) if logits_symbol is not None else torch.zeros(1, device=device)
    loss_name = criterion(logits_name, labels) if logits_name is not None else torch.zeros(1, device=device)
    total = ce_full_weight * loss_full + ce_symbol_weight * loss_symbol + ce_name_weight * loss_name
    return total, loss_full.detach(), loss_symbol.detach(), loss_name.detach()


def _run_pipeline_step(cmd: list[str], label: str) -> Tuple[float, int]:
    print(f"[PIPELINE] Starte {label}: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, check=False)
    duration = time.time() - start
    if result.returncode == 0:
        print(f"[PIPELINE] {label} abgeschlossen nach {duration:.1f}s ({duration/60:.2f} min)")
    else:
        print(f"[WARN] {label} fehlgeschlagen (Returncode {result.returncode}) nach {duration:.1f}s")
    return duration, result.returncode


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = get_training_config(cfg, "coarse")
    pipeline_cfg = cfg.get("training", {}).get("pipeline", {})
    auto_pipeline = bool(pipeline_cfg.get("auto_run_fine_and_export", False))
    val_cfg = train_cfg.get("validation", {})
    val_enabled = bool(val_cfg.get("enabled", False))
    val_split = float(val_cfg.get("split_ratio", 0.0))
    debug_cfg = cfg.get("debug", {})
    debug_enabled = bool(debug_cfg.get("enable", False))
    total_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Coarse-Training auf {device}")

    dataset = CoarseDataset(cfg)
    total_samples = len(dataset)
    print(f"[DATA] Karten={len(dataset.card_ids)} | Samples={total_samples}")
    preview_dir = debug_cfg.get("augmentation_preview_dir")
    if debug_enabled and preview_dir:
        print(f"[DEBUG] Schreibe Augmentierungs-Vorschau nach {preview_dir}")
        dataset.save_augmentations_of_first_card(preview_dir)

    train_dataset = dataset
    val_dataset = None

    if val_enabled and 0.0 < val_split < 1.0 and total_samples > 1:
        indices = list(range(total_samples))
        random.shuffle(indices)
        val_count = max(1, int(total_samples * val_split))
        if val_count >= total_samples:
            val_count = total_samples - 1
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]
        if train_indices and val_indices:
            train_dataset = DatasetSubset(dataset, train_indices)
            val_dataset = DatasetSubset(dataset, val_indices)
            print(
                f"[SPLIT] Train-Samples={len(train_dataset)} | Val-Samples={len(val_dataset)}"
            )
        else:
            print("[WARN] Val-Split: keine gueltigen Sample-Indizes gefunden, verwende kompletten Datensatz fuer Training.")

    batch_size = int(train_cfg.get("batch_size", 64))
    configured_workers = int(train_cfg.get("num_workers", 4))
    max_workers = max(1, os.cpu_count() or 2)
    num_workers = max(0, min(configured_workers, max_workers))
    if num_workers != configured_workers:
        print(f"[LOAD] num_workers von {configured_workers} auf {num_workers} angepasst (Systemlimit).")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    print(
        f"[LOAD] batch_size={batch_size} | Train-Schritte={len(train_loader)} "
        f"| Val-Schritte={len(val_loader) if val_loader else 0}"
    )

    model = build_card_encoder(cfg, num_classes=len(dataset.card_ids)).to(device)
    print(f"[MODEL] Backbone={cfg['encoder']['type']} | Klassen={len(dataset.card_ids)}")
    optimizer = Adam(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler = None
    if scheduler_cfg.get("use", False):
        sched_type = scheduler_cfg.get("type", "step").lower()
        if sched_type == "step":
            step_size = int(scheduler_cfg.get("step_size", 10))
            gamma = float(scheduler_cfg.get("gamma", 0.5))
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif sched_type == "cosine":
            t_max = int(scheduler_cfg.get("cosine_t_max", train_cfg.get("epochs", 30)))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    ce_weight_full = float(train_cfg.get("ce_full_weight", 1.0))
    ce_weight_symbol = float(train_cfg.get("ce_symbol_weight", 1.0))
    ce_weight_name = float(train_cfg.get("ce_name_weight", 1.0))
    ce_loss = nn.CrossEntropyLoss()

    debug_root = cfg.get("paths", {}).get("debug_dir", "./debug")
    log_dir = os.path.join(debug_root, "logs", "coarse")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[LOG] TensorBoard unter {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    best_loss = float("inf")
    best_state = None

    epochs = int(train_cfg.get("epochs", 1))
    coarse_start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_full = 0.0
        running_symbol = 0.0
        running_name = 0.0
        running_correct = 0
        running_samples = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            full_batch = batch[0].to(device, non_blocking=True)
            symbol_batch = batch[1].to(device, non_blocking=True)
            name_batch = batch[2].to(device, non_blocking=True)
            labels = batch[3].to(device, non_blocking=True)
            embeddings, logits = model(full_batch, symbol_batch, name_batch, return_logits=True)
            total_loss, loss_full, loss_symbol, loss_name = _compute_losses(
                logits, labels, ce_weight_full, ce_weight_symbol, ce_weight_name, ce_loss
            )
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_full += loss_full.item()
            running_symbol += loss_symbol.item()
            running_name += loss_name.item()
            logits_full = logits[0]
            if logits_full is not None:
                preds = logits_full.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
            running_samples += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        avg_full = running_full / len(train_loader)
        avg_symbol = running_symbol / len(train_loader)
        avg_name = running_name / len(train_loader)
        train_acc = running_correct / max(1, running_samples)
        writer.add_scalar("loss/total", avg_loss, epoch)
        writer.add_scalar("loss/full_ce", avg_full, epoch)
        writer.add_scalar("loss/symbol_ce", avg_symbol, epoch)
        writer.add_scalar("loss/name_ce", avg_name, epoch)
        writer.add_scalar("metrics/train_accuracy", train_acc, epoch)

        val_loss = None
        val_acc = None
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            v_correct = 0
            v_samples = 0
            with torch.no_grad():
                for full_batch, symbol_batch, name_batch, labels in val_loader:
                    full_batch = full_batch.to(device, non_blocking=True)
                    symbol_batch = symbol_batch.to(device, non_blocking=True)
                    name_batch = name_batch.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    _, logits = model(full_batch, symbol_batch, name_batch, return_logits=True)
                    ce_total, _, _, _ = _compute_losses(
                        logits, labels, ce_weight_full, ce_weight_symbol, ce_weight_name, ce_loss
                    )
                    v_loss += ce_total.item()

                    logits_full = logits[0]
                    if logits_full is not None:
                        preds = logits_full.argmax(dim=1)
                        v_correct += (preds == labels).sum().item()
                    v_samples += labels.size(0)

            val_loss = v_loss / len(val_loader)
            val_acc = v_correct / max(1, v_samples)
            writer.add_scalar("val/loss_ce", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            model.train()

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr/current", current_lr, epoch)

        print(
            f"[Epoch {epoch + 1}] total={avg_loss:.4f} fullCE={avg_full:.4f} symbolCE={avg_symbol:.4f} "
            f"nameCE={avg_name:.4f} trainAcc={train_acc:.4f}"
            + (
                f" | valCE={val_loss:.4f} valAcc={val_acc:.4f}"
                if val_loss is not None and val_acc is not None
                else ""
            )
            + f" | lr={current_lr:.6f}"
        )

        monitored = val_loss if val_loss is not None else avg_loss
        if monitored < best_loss:
            best_loss = monitored
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    writer.close()
    coarse_duration = time.time() - coarse_start_time
    print(f"[TIME] Coarse-Training Dauer: {coarse_duration:.1f}s ({coarse_duration/60:.2f} min)")

    if best_state is None:
        raise RuntimeError("Training fehlgeschlagen: keine validen Gewichte gefunden.")

    model.load_state_dict(best_state, strict=False)
    model.eval()

    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    out_path = os.path.join(cfg["paths"]["models_dir"], "encoder_coarse.pt")
    save_encoder(model, out_path, card_ids=dataset.card_ids)
    print(f"[INFO] Bestes Modell gespeichert unter {out_path}")
    print("[NEXT] Fine-Tuning starten: python -m src.training.train_fine --config config.yaml")
    if auto_pipeline:
        print("[PIPELINE] Automatischer Lauf aktiviert: Fine-Training + Embedding-Export")
        fine_cmd = [sys.executable, "-m", "src.training.train_fine", "--config", args.config]
        _, fine_rc = _run_pipeline_step(fine_cmd, "Fine-Training")
        if fine_rc == 0:
            export_cmd = [sys.executable, "-m", "src.training.export_embeddings", "--config", args.config]
            _run_pipeline_step(export_cmd, "Embedding-Export")
        else:
            print("[PIPELINE] Fine-Training fehlgeschlagen, Embedding-Export wird uebersprungen.")

    total_duration = time.time() - total_start
    print(f"[TIME] Gesamtzeit: {total_duration:.1f}s ({total_duration/60:.2f} min)")


if __name__ == "__main__":
    main()



