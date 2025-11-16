import argparse
import os
import sys
from pathlib import Path
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = get_training_config(cfg, "fine")

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
    print(f"[LOAD] batch_size={train_cfg.get('batch_size')} | Schritte pro Epoche={len(dataloader)}")

    coarse_path = os.path.join(cfg["paths"]["models_dir"], "encoder_coarse.pt")
    if not os.path.exists(coarse_path):
        raise FileNotFoundError(f"Coarse-Checkpoint nicht gefunden: {coarse_path}")

    model = load_encoder(coarse_path, cfg, num_classes=len(dataset.card_ids), device=device)
    model.num_classes = len(dataset.card_ids)
    print(f"[MODEL] Init von {coarse_path} | Klassen={model.num_classes}")

    freeze_ratio = float(train_cfg.get("freeze_ratio", 0.6))
    _freeze_model(model, freeze_ratio)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=float(train_cfg.get("lr", 1e-4)))
    triplet_loss = nn.TripletMarginLoss(margin=float(train_cfg.get("margin", 0.35)), p=2)
    ce_loss = nn.CrossEntropyLoss()

    debug_root = cfg.get("paths", {}).get("debug_dir", "./debug")
    log_dir = os.path.join(debug_root, "logs", "fine")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[LOG] TensorBoard unter {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    best_loss = float("inf")
    best_state = None

    epochs = int(train_cfg.get("epochs", 1))
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
            total_loss = loss_triplet + loss_ce_total
            total_loss.backward()
            optimizer.step()

            running_triplet += loss_triplet.item()
            running_ce += loss_ce_total.item()

        avg_triplet = running_triplet / len(dataloader)
        avg_ce = running_ce / len(dataloader)
        total = avg_triplet + avg_ce

        writer.add_scalar("loss/triplet", avg_triplet, epoch)
        writer.add_scalar("loss/ce", avg_ce, epoch)
        writer.add_scalar("loss/total", total, epoch)
        print(f"[Epoch {epoch + 1}] total={total:.4f} triplet={avg_triplet:.4f} CE={avg_ce:.4f}")

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
    print("[NEXT] Embeddings exportieren: python -m src.training.export_embeddings --config config.yaml")


if __name__ == "__main__":
    main()


