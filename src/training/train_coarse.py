import argparse
import os
import sys
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
from src.datasets.card_datasets import CoarseDataset
from src.core.model_builder import build_encoder_model, save_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coarse-Training des Karten-Encoders.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Pfad zur Konfigurationsdatei")
    return parser.parse_args()


def _build_dataloader(dataset: CoarseDataset, train_cfg: dict) -> DataLoader:
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", min(8, os.cpu_count() or 2)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _compute_losses(
    logits: Tuple[torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    ce_full_weight: float,
    ce_symbol_weight: float,
    criterion: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_full, logits_symbol = logits
    loss_full = criterion(logits_full, labels) if logits_full is not None else torch.zeros(1, device=labels.device)
    loss_symbol = criterion(logits_symbol, labels) if logits_symbol is not None else torch.zeros(1, device=labels.device)
    total = ce_full_weight * loss_full + ce_symbol_weight * loss_symbol
    return total, loss_full.detach(), loss_symbol.detach()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = get_training_config(cfg, "coarse")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Coarse-Training auf {device}")

    dataset = CoarseDataset(cfg)
    print(f"[DATA] Karten={len(dataset.card_ids)} | Samples={len(dataset)}")
    preview_dir = cfg.get("debug", {}).get("augmentation_preview_dir")
    if preview_dir:
        print(f"[DEBUG] Schreibe Augmentierungs-Vorschau nach {preview_dir}")
        dataset.save_augmentations_of_first_card(preview_dir)

    dataloader = _build_dataloader(dataset, train_cfg)
    print(f"[LOAD] batch_size={train_cfg.get('batch_size')} | Schritte pro Epoche={len(dataloader)}")

    model = build_encoder_model(cfg, num_classes=len(dataset.card_ids)).to(device)
    print(f"[MODEL] Backbone={cfg['encoder']['type']} | Klassen={len(dataset.card_ids)}")
    optimizer = Adam(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    ce_weight_full = float(train_cfg.get("ce_full_weight", 1.0))
    ce_weight_symbol = float(train_cfg.get("ce_symbol_weight", 1.0))
    ce_loss = nn.CrossEntropyLoss()

    debug_root = cfg.get("paths", {}).get("debug_dir", "./debug")
    log_dir = os.path.join(debug_root, "logs", "coarse")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[LOG] TensorBoard unter {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    best_loss = float("inf")
    best_state = None

    epochs = int(train_cfg.get("epochs", 1))
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_full = 0.0
        running_symbol = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            full_batch, symbol_batch, labels = [b.to(device) for b in batch]
            embeddings, logits = model(full_batch, symbol_batch, return_logits=True)
            total_loss, loss_full, loss_symbol = _compute_losses(
                logits, labels, ce_weight_full, ce_weight_symbol, ce_loss
            )
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_full += loss_full.item()
            running_symbol += loss_symbol.item()

        avg_loss = running_loss / len(dataloader)
        avg_full = running_full / len(dataloader)
        avg_symbol = running_symbol / len(dataloader)
        writer.add_scalar("loss/total", avg_loss, epoch)
        writer.add_scalar("loss/full_ce", avg_full, epoch)
        writer.add_scalar("loss/symbol_ce", avg_symbol, epoch)
        print(f"[Epoch {epoch + 1}] total={avg_loss:.4f} fullCE={avg_full:.4f} symbolCE={avg_symbol:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    writer.close()

    if best_state is None:
        raise RuntimeError("Training fehlgeschlagen: keine validen Gewichte gefunden.")

    model.load_state_dict(best_state, strict=False)
    model.eval()

    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    out_path = os.path.join(cfg["paths"]["models_dir"], "encoder_coarse.pt")
    save_encoder(model, out_path, card_ids=dataset.card_ids)
    print(f"[INFO] Bestes Modell gespeichert unter {out_path}")
    print("[NEXT] Fine-Tuning starten: python -m src.training.train_fine --config config.yaml")


if __name__ == "__main__":
    main()


