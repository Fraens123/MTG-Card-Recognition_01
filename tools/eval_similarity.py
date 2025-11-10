#!/usr/bin/env python3
"""
Auswertungsskript: misst Cosine-Similarity-Gap und Top-1-Genauigkeit
auf Basis der in cards.json gespeicherten Embeddings.
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluiert Embedding-Qualität anhand cards.json.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/cards.json"),
        help="Pfad zur Karten-Datenbank (Default: data/cards.json)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Optional: zufällige Anzahl von Embeddings evaluieren (beschleunigt große DBs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed für die Stichprobe mit --sample.",
    )
    return parser.parse_args()


def load_database(db_path: Path) -> tuple[torch.Tensor, list[str]]:
    if not db_path.exists():
        raise FileNotFoundError(f"cards.json nicht gefunden: {db_path}")
    with db_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    embeddings = data.get("embeddings")
    cards = data.get("cards")
    if not embeddings or not cards:
        raise ValueError(f"{db_path} enthält keine 'embeddings' oder 'cards'.")
    if len(embeddings) != len(cards):
        raise ValueError(
            f"Anzahl Embeddings ({len(embeddings)}) != Anzahl Karten ({len(cards)}). "
            "Bitte generate_embeddings.py erneut ausführen."
        )
    tensor = torch.tensor(embeddings, dtype=torch.float32)
    labels = [card.get("card_uuid", f"card_{idx}") for idx, card in enumerate(cards)]
    return tensor, labels


def select_subset(tensor: torch.Tensor, labels: list[str], sample_size: int, seed: int) -> tuple[torch.Tensor, list[str]]:
    if sample_size is None or sample_size >= tensor.shape[0]:
        return tensor, labels
    rng = random.Random(seed)
    indices = list(range(tensor.shape[0]))
    rng.shuffle(indices)
    subset = indices[:sample_size]
    subset.sort()
    tensor = tensor[subset]
    labels = [labels[i] for i in subset]
    return tensor, labels


def evaluate_embeddings(embeddings: torch.Tensor, labels: list[str]) -> dict:
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    sims = embeddings @ embeddings.T
    sims.fill_diagonal_(-1.0)

    unique_labels = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    label_ids = torch.tensor([unique_labels[label] for label in labels], dtype=torch.long)

    same_mask = label_ids.unsqueeze(0) == label_ids.unsqueeze(1)
    diff_mask = ~same_mask
    same_mask.fill_diagonal_(False)

    pos_sims = sims.masked_fill(~same_mask, -1.0)
    neg_sims = sims.masked_fill(~diff_mask, -1.0)

    best_pos, pos_indices = pos_sims.max(dim=1)
    best_neg, _ = neg_sims.max(dim=1)

    valid_pos_mask = best_pos > 0
    if valid_pos_mask.any():
        top1_matches = label_ids[pos_indices] == label_ids
        top1_acc = top1_matches[valid_pos_mask].float().mean().item()
        avg_pos = best_pos[valid_pos_mask].mean().item()
        avg_gap = (best_pos[valid_pos_mask] - best_neg[valid_pos_mask]).mean().item()
    else:
        top1_acc = float("nan")
        avg_pos = float("nan")
        avg_gap = float("nan")

    avg_neg = best_neg.mean().item()

    return {
        "embeddings": embeddings.shape[0],
        "classes": len(unique_labels),
        "top1_acc": top1_acc,
        "avg_positive_sim": avg_pos,
        "avg_hard_negative_sim": avg_neg,
        "avg_similarity_gap": avg_gap,
        "valid_positive_pairs": int(valid_pos_mask.sum().item()),
    }


def main() -> None:
    args = parse_args()
    embeddings, labels = load_database(args.db)
    embeddings, labels = select_subset(embeddings, labels, args.sample, args.seed)
    stats = evaluate_embeddings(embeddings, labels)
    print("=== Embedding Evaluation ===")
    print(f"Embeddings insgesamt: {stats['embeddings']} (Klassen: {stats['classes']})")
    print(f"Gültige Positive pro Karte: {stats['valid_positive_pairs']}")
    print(f"Durchschnittliche Positiv-Similarity: {stats['avg_positive_sim']:.3f}")
    print(f"Härteste Negativ-Similarity (Durchschnitt): {stats['avg_hard_negative_sim']:.3f}")
    print(f"Ø Similarity-Gap (Pos - Neg): {stats['avg_similarity_gap']:.3f}")
    if stats["valid_positive_pairs"] > 0:
        print(f"Top-1-Accuracy (nur Karten mit >1 Embedding): {stats['top1_acc']:.3f}")
    else:
        print("Top-1-Accuracy: nicht berechenbar (jede Karte hat nur ein Embedding).")


if __name__ == "__main__":
    main()
