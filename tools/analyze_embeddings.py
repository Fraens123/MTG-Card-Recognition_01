from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse card embeddings for intra-/inter-card cosine similarities."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config that points to the embedding database.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Direct path to the embeddings JSON database (overrides config).",
    )
    parser.add_argument(
        "--diff-samples",
        type=int,
        default=10_000,
        help="Number of random inter-card pairs to sample for similarity stats.",
    )
    parser.add_argument(
        "--table-size",
        type=int,
        default=10,
        help="How many cards with the lowest intra-card mean similarity to print.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling/TSNE.",
    )
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Enable t-SNE visualisation of a subset of embeddings.",
    )
    parser.add_argument(
        "--tsne-samples",
        type=int,
        default=500,
        help="How many embeddings to sample when running t-SNE.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Perplexity value used for t-SNE (should be < sample count).",
    )
    return parser.parse_args()


def load_embeddings(config_path: str, explicit_db: str | None) -> tuple[Path, List[Dict], np.ndarray]:
    cfg = load_config(config_path)
    db_path = Path(explicit_db or cfg.get("database", {}).get("path", "./data/cards.json"))
    if not db_path.exists():
        raise FileNotFoundError(f"Database JSON not found: {db_path}")

    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cards: List[Dict] = data.get("cards", [])
    embeddings = np.asarray(data.get("embeddings", []), dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings array must be 2D [N, D].")
    if len(cards) != len(embeddings):
        raise ValueError("cards and embeddings must have the same length.")
    return db_path, cards, embeddings


def compute_intra_card_stats(by_name: Dict[str, List[np.ndarray]]) -> tuple[List[Dict], np.ndarray]:
    stats: List[Dict] = []
    all_pair_sims: List[np.ndarray] = []
    for name, vectors in by_name.items():
        if len(vectors) < 2:
            continue
        stacked = np.stack(vectors)
        sim_matrix = cosine_similarity(stacked)
        tri = np.triu_indices_from(sim_matrix, k=1)
        pair_sims = sim_matrix[tri]
        if pair_sims.size == 0:
            continue
        stats.append(
            {
                "name": name,
                "count": stacked.shape[0],
                "mean": float(np.mean(pair_sims)),
                "min": float(np.min(pair_sims)),
                "max": float(np.max(pair_sims)),
            }
        )
        all_pair_sims.append(pair_sims)

    combined = np.concatenate(all_pair_sims) if all_pair_sims else np.array([], dtype=np.float32)
    return stats, combined


def sample_inter_card_similarities(
    normalized_embeddings: np.ndarray,
    names: Sequence[str],
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total = len(names)
    if total < 2:
        return np.array([], dtype=np.float32)

    sims: List[float] = []
    max_attempts = num_samples * 20
    attempts = 0
    while len(sims) < num_samples and attempts < max_attempts:
        i = rng.integers(0, total)
        j = rng.integers(0, total)
        attempts += 1
        if i == j or names[i] == names[j]:
            continue
        sims.append(float(np.dot(normalized_embeddings[i], normalized_embeddings[j])))
    return np.asarray(sims, dtype=np.float32)


def print_worst_cards(card_stats: List[Dict], table_size: int):
    if not card_stats:
        print("No cards with >= 2 embeddings found; skipping intra-card table.")
        return

    sorted_stats = sorted(card_stats, key=lambda x: x["mean"])
    print("\nCards with lowest intra-card mean cosine similarity:")
    header = f"{'Card name':60} {'#Emb':>5} {'mean':>8} {'min':>8} {'max':>8}"
    print(header)
    print("-" * len(header))
    for entry in sorted_stats[:table_size]:
        name = entry["name"]
        print(
            f"{name[:60]:60} "
            f"{entry['count']:>5d} "
            f"{entry['mean']:>8.4f} "
            f"{entry['min']:>8.4f} "
            f"{entry['max']:>8.4f}"
        )


def plot_similarity_histograms(
    intra_values: np.ndarray,
    inter_values: np.ndarray,
    intra_mean: float | None,
    inter_mean: float | None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    bins = np.linspace(0.0, 1.0, 41)

    def _plot_hist(ax, values: np.ndarray, title: str, mean_value: float | None):
        if values.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(values, bins=bins, color="#1f77b4", alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel("Cosine similarity")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        if mean_value is not None:
            ax.axvline(mean_value, color="red", linestyle="--", label=f"mean={mean_value:.3f}")
            ax.legend()

    _plot_hist(axes[0], intra_values, "Intra-card cosine similarities", intra_mean)
    _plot_hist(axes[1], inter_values, "Inter-card cosine similarities", inter_mean)
    axes[0].set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def maybe_run_tsne(
    embeddings: np.ndarray,
    names: Sequence[str],
    sample_size: int,
    perplexity: float,
    rng: np.random.Generator,
):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("sklearn.manifold.TSNE not available; skipping t-SNE plot.")
        return

    total = len(embeddings)
    if total < 2:
        print("Not enough embeddings for t-SNE.")
        return

    sample_size = min(sample_size, total)
    if sample_size < 2:
        print("Need at least 2 samples for t-SNE.")
        return
    if perplexity >= sample_size:
        perplexity = max(5.0, sample_size / 3)
        print(f"Adjusted t-SNE perplexity to {perplexity:.1f} to stay < sample size.")

    indices = rng.choice(total, size=sample_size, replace=False)
    sampled_embeddings = embeddings[indices]
    sampled_names = np.array(names)[indices]

    print(f"Running t-SNE on {sample_size} embeddings ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=int(rng.integers(0, 1_000_000)),
        init="pca",
        learning_rate="auto",
    )
    points = tsne.fit_transform(sampled_embeddings)

    unique_names = list(dict.fromkeys(sampled_names))
    cmap = plt.cm.get_cmap("tab20", len(unique_names))

    plt.figure(figsize=(8, 6))
    for idx, card_name in enumerate(unique_names):
        mask = sampled_names == card_name
        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            s=30,
            alpha=0.8,
            color=cmap(idx),
            label=f"{card_name} ({mask.sum()})",
        )
    plt.title("t-SNE of card embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    db_path, cards, embeddings = load_embeddings(args.config, args.db_path)
    names = [card.get("name", "unknown") for card in cards]

    num_embeddings, dimension = embeddings.shape
    unique_names = len(set(names))

    print(f"Database path: {db_path}")
    print(f"Total embeddings: {num_embeddings}")
    print(f"Embedding dimension: {dimension}")
    print(f"Unique card names: {unique_names}")

    by_name: Dict[str, List[np.ndarray]] = defaultdict(list)
    for emb, card in zip(embeddings, cards):
        by_name[card.get("name", "unknown")].append(emb)

    card_stats, intra_pair_sims = compute_intra_card_stats(by_name)
    if card_stats:
        mean_same = float(np.mean([entry["mean"] for entry in card_stats]))
        min_same = float(np.min([entry["min"] for entry in card_stats]))
        max_same = float(np.max([entry["max"] for entry in card_stats]))
        print(
            f"Intra-card mean similarity (averaged over cards): {mean_same:.4f} | "
            f"min of mins: {min_same:.4f} | max of maxes: {max_same:.4f}"
        )
    else:
        mean_same = min_same = max_same = None
        print("No intra-card similarity stats available (need >=2 embeddings per card).")

    print_worst_cards(card_stats, args.table_size)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.clip(norms, 1e-12, None)
    normalized_embeddings = embeddings / safe_norms
    inter_sims = sample_inter_card_similarities(
        normalized_embeddings, names, args.diff_samples, rng
    )

    if inter_sims.size > 0:
        mean_inter = float(np.mean(inter_sims))
        max_inter = float(np.max(inter_sims))
        perc95 = float(np.percentile(inter_sims, 95))
        print(
            f"Inter-card similarities (different names): mean={mean_inter:.4f}, "
            f"max={max_inter:.4f}, 95th percentile={perc95:.4f}"
        )
    else:
        mean_inter = max_inter = perc95 = None
        print("Not enough distinct card pairs to compute inter-card similarities.")

    plot_similarity_histograms(
        intra_pair_sims, inter_sims, mean_same if mean_same is not None else None, mean_inter
    )

    if args.tsne:
        maybe_run_tsne(embeddings, names, args.tsne_samples, args.tsne_perplexity, rng)


if __name__ == "__main__":
    main()
