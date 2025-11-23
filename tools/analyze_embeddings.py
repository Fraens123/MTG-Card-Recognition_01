from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_utils import load_config
from src.core.sqlite_store import load_embeddings_with_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse card embeddings for intra-/inter-card cosine similarities."
    )
    parser.add_argument(
        "--mode",
        choices=["runtime", "analysis"],
        default="analysis",
        help="Welchen Export-Modus aus der config laden (default: analysis).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config that points to the embedding database.",
    )
    parser.add_argument(
        "--database",
        "--db-path",
        dest="database",
        type=str,
        default=None,
        help="Direct path to the SQLite embeddings DB (overrides config).",
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


def load_embeddings_and_config(
    config_path: str | None, explicit_db: str | None, mode: str = "analysis"
) -> tuple[Path, Dict, List[Dict], np.ndarray, float | None]:
    """
    Laedt Embeddings + optional Config. Gibt den Pfad zur DB, Config-Dict,
    Kartenliste, Embedding-Array und optional den Recognition-Threshold zurueck.
    """
    if not config_path and not explicit_db:
        default_cfg = Path("config.yaml")
        if default_cfg.exists():
            config_path = str(default_cfg)
        else:
            raise ValueError("Bitte --config oder --database angeben.")

    cfg: Dict = {}
    threshold: float | None = None
    emb_dim: int = 1024
    if config_path:
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config nicht gefunden: {cfg_path}")
        cfg = load_config(config_path)
        threshold = cfg.get("recognition", {}).get("threshold")
        emb_dim = int(cfg.get("encoder", {}).get("emb_dim") or cfg.get("model", {}).get("embed_dim", emb_dim))

    if explicit_db:
        db_path = Path(explicit_db)
    else:
        db_default = cfg.get("database", {}).get("sqlite_path") or "tcg_database/database/karten.db"
        db_path = Path(db_default)

    if not db_path.exists():
        raise FileNotFoundError(f"Database nicht gefunden: {db_path}")

    scenario = cfg.get("database", {}).get("scenario") or "default"

    embeddings_by_card, meta_by_card = load_embeddings_with_meta(
        str(db_path), mode, emb_dim, scenario=scenario
    )
    cards: List[Dict] = []
    vectors: List[np.ndarray] = []
    for cid, vecs in embeddings_by_card.items():
        meta = meta_by_card.get(cid, {})
        base = {
            "card_uuid": cid,
            "name": meta.get("name") or cid,
            "set_code": meta.get("set") or "",
            "collector_number": meta.get("collector_number") or "",
            "image_path": (meta.get("image_paths") or [None])[0] or "",
        }
        for vec in vecs:
            cards.append(base.copy())
            vectors.append(vec)

    embeddings = np.stack(vectors, axis=0) if vectors else np.zeros((0, emb_dim), dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings array must be 2D [N, D].")
    if len(cards) != len(embeddings):
        raise ValueError("cards and embeddings must have the same length.")
    return db_path, cfg, cards, embeddings, threshold


def group_embeddings_by_card(cards: List[Dict], embeddings: np.ndarray) -> tuple[Dict[str, List[np.ndarray]], Dict[str, Dict]]:
    """
    Gruppiert Embeddings nach card_id (card_uuid foerderlich), faellt auf name zurueck.
    """
    by_card: Dict[str, List[np.ndarray]] = defaultdict(list)
    meta: Dict[str, Dict] = {}
    for emb, card in zip(embeddings, cards):
        card_id = card.get("card_uuid") or card.get("name") or card.get("id") or "unknown"
        by_card[card_id].append(emb)
        meta[card_id] = card
    return by_card, meta


def _pairwise_cosine(values: np.ndarray) -> np.ndarray:
    """Berechnet obere Dreieck-Cosine-Sims fuer normierte Vektoren."""
    sims = values @ values.T
    tri = np.triu_indices_from(sims, k=1)
    return sims[tri]


def compute_intra_card_metrics(
    groups: Dict[str, List[np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, Dict[str, float], List[Dict]]:
    intra_cos: List[np.ndarray] = []
    per_card_stats: List[Dict] = []
    for card_id, vecs in groups.items():
        if len(vecs) < 2:
            continue
        stacked = np.stack(vecs)
        cos_vals = _pairwise_cosine(stacked)
        if cos_vals.size == 0:
            continue
        dist_vals = 1.0 - cos_vals
        per_card_stats.append(
            {
                "card_id": card_id,
                "count": stacked.shape[0],
                "mean_cos": float(np.mean(cos_vals)),
                "mean_dist": float(np.mean(dist_vals)),
            }
        )
        intra_cos.append(cos_vals)

    cos_concat = np.concatenate(intra_cos) if intra_cos else np.array([], dtype=np.float32)
    dist_concat = 1.0 - cos_concat if cos_concat.size else np.array([], dtype=np.float32)

    stats = {
        "mean_cos": float(np.mean(cos_concat)) if cos_concat.size else float("nan"),
        "mean_dist": float(np.mean(dist_concat)) if dist_concat.size else float("nan"),
        "median_dist": float(np.median(dist_concat)) if dist_concat.size else float("nan"),
        "std_dist": float(np.std(dist_concat)) if dist_concat.size else float("nan"),
    }
    return cos_concat, dist_concat, stats, per_card_stats


def _compute_centroids(groups: Dict[str, List[np.ndarray]]) -> tuple[np.ndarray, List[str]]:
    ids: List[str] = []
    centroids: List[np.ndarray] = []
    for card_id, vecs in groups.items():
        stacked = np.stack(vecs)
        centroid = np.mean(stacked, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        ids.append(card_id)
        centroids.append(centroid)
    if not centroids:
        return np.empty((0, 0), dtype=np.float32), ids
    return np.stack(centroids), ids


def compute_inter_card_metrics(
    groups: Dict[str, List[np.ndarray]],
    rng: np.random.Generator,
    max_cards: int = 8000,
) -> tuple[np.ndarray, np.ndarray, Dict[str, float], np.ndarray, List[str]]:
    """
    Nutzt zentroidenbasierte Inter-Card-Distanzen.
    Bei sehr vielen Karten wird auf max_cards gecuttet (random subset) aus Performance-Gruenden.
    """
    centroids, card_ids = _compute_centroids(groups)
    total_cards = len(card_ids)
    if total_cards < 2:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            {"mean_cos": float("nan"), "mean_dist": float("nan"), "median_dist": float("nan"), "std_dist": float("nan")},
            centroids,
            card_ids,
        )

    if total_cards > max_cards:
        idx = rng.choice(total_cards, size=max_cards, replace=False)
        centroids = centroids[idx]
        card_ids = [card_ids[i] for i in idx]

    sim_matrix = centroids @ centroids.T  # cos bei normierten Zentroiden
    np.fill_diagonal(sim_matrix, 0.0)
    tri = np.triu_indices_from(sim_matrix, k=1)
    cos_vals = sim_matrix[tri]
    dist_vals = 1.0 - cos_vals
    stats = {
        "mean_cos": float(np.mean(cos_vals)),
        "mean_dist": float(np.mean(dist_vals)),
        "median_dist": float(np.median(dist_vals)),
        "std_dist": float(np.std(dist_vals)),
    }
    return cos_vals, dist_vals, stats, centroids, card_ids


def compute_glide_quality(D_aug: float, Dist_med: float) -> float:
    return float(Dist_med / D_aug) if D_aug and D_aug == D_aug else float("nan")


def summarize_intra_variance(per_card_stats: List[Dict], meta: Dict[str, Dict], top_n: int = 20):
    if not per_card_stats:
        print("[WARN] Keine Karten mit >=2 Embeddings fuer Intra-Analyse.")
        return
    sorted_cards = sorted(per_card_stats, key=lambda x: x["mean_dist"], reverse=True)
    print("\n[WARN] Karten mit hoher Augmentierungs-Varianz (Intra-Card-Problem):")
    for idx, entry in enumerate(sorted_cards[:top_n], start=1):
        card = meta.get(entry["card_id"], {})
        name = card.get("name", entry["card_id"])
        print(f"  {idx:2d}) {name} (card_id={entry['card_id']}, mean_intra_dist={entry['mean_dist']:.4f}, n={entry['count']})")


def summarize_confusable_cards(
    centroids: np.ndarray, card_ids: List[str], meta: Dict[str, Dict], top_n: int = 20
):
    if len(card_ids) < 2:
        print("[WARN] Zu wenige Karten fuer Inter-Card-Problem-Liste.")
        return
    sim_matrix = centroids @ centroids.T
    np.fill_diagonal(sim_matrix, -np.inf)  # self ignore
    nearest = np.argmax(sim_matrix, axis=1)
    nearest_cos = sim_matrix[np.arange(len(card_ids)), nearest]
    nearest_dist = 1.0 - nearest_cos
    records = []
    for idx, card_id in enumerate(card_ids):
        neighbor_idx = int(nearest[idx])
        neighbor_id = card_ids[neighbor_idx]
        records.append(
            {
                "card_id": card_id,
                "neighbor_id": neighbor_id,
                "cos": float(nearest_cos[idx]),
                "dist": float(nearest_dist[idx]),
            }
        )
    records.sort(key=lambda x: x["dist"])  # kleinste Distanz = kritisch
    print("\n[WARN] Karten mit sehr ähnlichen Nachbarn (Inter-Card-Problem):")
    for idx, rec in enumerate(records[:top_n], start=1):
        name_a = meta.get(rec["card_id"], {}).get("name", rec["card_id"])
        name_b = meta.get(rec["neighbor_id"], {}).get("name", rec["neighbor_id"])
        print(f"  {idx:2d}) {name_a} ~ {name_b} (cos={rec['cos']:.4f}, dist={rec['dist']:.4f})")


def plot_similarity_histograms(
    intra_values: np.ndarray,
    inter_values: np.ndarray,
    intra_mean: float | None,
    inter_mean: float | None,
    D_aug: float | None,
    Dist_med: float | None,
    Q: float | None,
    threshold: float | None,
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
        if mean_value is not None and mean_value == mean_value:  # not NaN
            ax.axvline(mean_value, color="red", linestyle="--", label=f"mean={mean_value:.3f}")
        if threshold is not None:
            ax.axvline(threshold, color="green", linestyle=":", label=f"thr={threshold:.2f}")
        if len(ax.get_lines()) > 0:
            ax.legend()

    _plot_hist(axes[0], intra_values, "Intra-card cosine similarities", intra_mean)
    _plot_hist(axes[1], inter_values, "Inter-card cosine similarities", inter_mean)
    axes[0].set_ylabel("Frequency")

    if D_aug is not None and Dist_med is not None and Q is not None:
        text = f"D_aug={D_aug:.3f}\nDist_med={Dist_med:.3f}\nQ={Q:.2f}"
        axes[0].text(
            0.05,
            0.95,
            text,
            transform=axes[0].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )

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

    db_path, cfg, cards, embeddings, cfg_threshold = load_embeddings_and_config(args.config, args.database, args.mode)
    print(f"[LOAD] Embedding-DB ({args.mode}): {db_path}")
    if embeddings.size == 0:
        raise SystemExit("Keine Embeddings gefunden.")
    # L2-Normalisierung absichern (Embeddings sollten schon normiert sein)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    by_card, meta = group_embeddings_by_card(cards, embeddings)
    num_embeddings, dimension = embeddings.shape
    total_cards = len(by_card)
    cards_with_multi = sum(1 for v in by_card.values() if len(v) >= 2)
    names = [c.get("name", "unknown") for c in cards]

    intra_cos, intra_dist, intra_stats, per_card_stats = compute_intra_card_metrics(by_card)
    inter_cos, inter_dist, inter_stats, centroids, centroid_ids = compute_inter_card_metrics(by_card, rng)

    D_aug = intra_stats["mean_dist"]
    Dist_med = inter_stats["median_dist"]
    Q = compute_glide_quality(D_aug, Dist_med)

    print("=" * 60)
    print("[QUALITY] Glide-Style Embedding-Analyse")
    print("=" * 60)
    print(f"Karten (>=1 Embedding):          {total_cards}")
    print(f"Karten (>=2 Embeddings):         {cards_with_multi}")
    print(f"Total embeddings:                {num_embeddings}")
    print(f"Embedding dimension:             {dimension}")

    print("\nIntra-Card (gleiche Karte):")
    print(f"  mean(cos)   = {intra_stats['mean_cos']:.4f}")
    print(f"  mean(dist)  = {intra_stats['mean_dist']:.4f}")
    print(f"  median(dist)= {intra_stats['median_dist']:.4f}")
    print(f"  std(dist)   = {intra_stats['std_dist']:.4f}")

    print("\nInter-Card (verschiedene Karten, Zentroiden):")
    print(f"  mean(cos)   = {inter_stats['mean_cos']:.4f}")
    print(f"  mean(dist)  = {inter_stats['mean_dist']:.4f}")
    print(f"  median(dist)= {inter_stats['median_dist']:.4f}")
    print(f"  std(dist)   = {inter_stats['std_dist']:.4f}")

    print("\nGlide-Style Kennzahlen:")
    print(f"  D_aug    = {D_aug:.4f}")
    print(f"  Dist_med = {Dist_med:.4f}")
    print(f"  Q = Dist_med / D_aug = {Q:.2f}")

    thr_cfg = cfg_threshold
    thr_low = 1.0 - Dist_med if Dist_med == Dist_med else float("nan")
    thr_high = 1.0 - (D_aug * 1.5) if D_aug == D_aug else float("nan")
    print("\nEmpfehlung (Heuristik):")
    print(f"  Cosine-Threshold ~ zwischen {thr_high:.3f} und {thr_low:.3f}")
    if thr_cfg is not None:
        print(f"  Aktueller Threshold aus config: {thr_cfg:.3f}")

    summarize_intra_variance(per_card_stats, meta, top_n=args.table_size)
    summarize_confusable_cards(centroids, centroid_ids, meta, top_n=args.table_size)

    plot_similarity_histograms(
        intra_cos,
        inter_cos,
        intra_stats["mean_cos"],
        inter_stats["mean_cos"],
        D_aug,
        Dist_med,
        Q,
        thr_cfg,
    )

    if args.tsne:
        maybe_run_tsne(embeddings, names, args.tsne_samples, args.tsne_perplexity, rng)


if __name__ == "__main__":
    main()
