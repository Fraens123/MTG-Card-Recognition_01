#!/usr/bin/env python3
"""
MTG Card Recognition CLI

Vergleicht Pi-Cam-Bilder mit der Scryfall-Datenbank anhand von Cosine-Similarities,
erstellt Vergleichsplots und fasst die Ergebnisse statistisch zusammen.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from .model import load_encoder
    from .db import SimpleCardDB
    from .image_pipeline import (
        build_resize_normalize_transform,
        crop_set_symbol,
        get_set_symbol_crop_cfg,
        resolve_resize_hw,
    )
    from .embedding_utils import build_card_embedding
except ImportError:
    from src.cardscanner.model import load_encoder
    from src.cardscanner.db import SimpleCardDB
    from src.cardscanner.image_pipeline import (
        build_resize_normalize_transform,
        crop_set_symbol,
        get_set_symbol_crop_cfg,
        resolve_resize_hw,
    )
    from src.cardscanner.embedding_utils import build_card_embedding


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config-Datei nicht gefunden: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data:
        raise ValueError(f"Config-Datei ist leer: {config_path}")
    return data


def get_inference_transforms(resize_hw: Tuple[int, int]) -> T.Compose:
    """Identische Resize/Normalize-Pipeline wie beim Embedding-Export."""
    return build_resize_normalize_transform(resize_hw)


def create_comparison_plot(
    camera_img_path: str,
    camera_img: Image.Image,
    best_match: Dict,
    similarity_score: float,
    search_time: float,
    output_path: str,
) -> None:
    """Erstellt eine Side-by-Side-Abbildung zwischen Kameraaufnahme und Scryfall-Referenz."""
    import matplotlib

    try:
        matplotlib.rcParams["font.family"] = "Noto Sans Symbols"
    except Exception:
        pass

    plt.style.use("default")
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.3,
        wspace=0.2,
        height_ratios=[0.1, 0.8, 0.1],
        width_ratios=[1, 1],
    )

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.5,
        "MTG Card Recognition - Similarity Search Results",
        fontsize=20,
        fontweight="bold",
        ha="center",
        va="center",
    )

    camera_ax = fig.add_subplot(gs[1, 0])
    camera_ax.imshow(camera_img)
    camera_ax.set_title("Pi Camera Image", fontsize=16, fontweight="bold", pad=20)
    camera_ax.axis("off")

    scryfall_ax = fig.add_subplot(gs[1, 1])
    if os.path.exists(best_match["image_path"]):
        ref_img = Image.open(best_match["image_path"]).convert("RGB")
        scryfall_ax.imshow(ref_img)
    else:
        scryfall_ax.text(
            0.5,
            0.5,
            "Referenzbild fehlt",
            ha="center",
            va="center",
            fontsize=14,
            transform=scryfall_ax.transAxes,
        )
    scryfall_ax.set_title("Best Match (Scryfall)", fontsize=16, fontweight="bold", pad=20)
    scryfall_ax.axis("off")

    info_ax = fig.add_subplot(gs[2, :])
    info_ax.axis("off")

    camera_filename = Path(camera_img_path).name
    info_text = (
        f"SEARCH RESULTS:\n"
        f"* Camera File: {camera_filename}\n"
        f"* Best Match: {best_match['name']} (Set: {best_match['set_code']}, #{best_match['collector_number']})\n"
        f"* Cosine Similarity: {similarity_score:.4f} ({similarity_score*100:.1f}%)\n"
        f"* Search Time: {search_time*1000:.2f} ms\n"
        f"* Card UUID: {best_match['card_uuid'][:8]}...\n"
        f"* Image Path: {Path(best_match['image_path']).name}"
    )

    if similarity_score > 0.8:
        color, status = ("green", "EXCELLENT MATCH")
    elif similarity_score > 0.6:
        color, status = ("orange", "GOOD MATCH")
    elif similarity_score > 0.4:
        color, status = ("red", "POOR MATCH")
    else:
        color, status = ("darkred", "NO MATCH")

    info_ax.text(
        0.02,
        0.95,
        info_text,
        fontsize=12,
        ha="left",
        va="top",
        transform=info_ax.transAxes,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )
    info_ax.text(
        0.98,
        0.95,
        status,
        fontsize=14,
        fontweight="bold",
        ha="right",
        va="top",
        transform=info_ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8, edgecolor="black"),
    )

    bar_width = 0.6
    bar_x = 0.2
    bar_y = 0.2
    info_ax.add_patch(
        patches.Rectangle(
            (bar_x, bar_y),
            bar_width,
            0.15,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=2,
        )
    )
    info_ax.add_patch(
        patches.Rectangle(
            (bar_x, bar_y),
            bar_width * similarity_score,
            0.15,
            facecolor=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
        )
    )
    info_ax.text(
        bar_x + bar_width / 2,
        bar_y + 0.075,
        f"{similarity_score:.3f}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="white",
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[PLOT] Vergleichsgrafik gespeichert: {output_path}")


def search_camera_image(
    model: torch.nn.Module,
    transform: T.Compose,
    crop_transform: T.Compose,
    db: SimpleCardDB,
    camera_img_path: str,
    device: torch.device,
    config: dict,
    crop_cfg: Optional[dict],
) -> Tuple[Optional[Dict], float, float]:
    """Berechnet das Query-Embedding und sucht den besten Match in der Datenbank."""
    start_time = time.time()

    with Image.open(camera_img_path) as img:
        rgb = img.convert("RGB")
        full_tensor = transform(rgb).unsqueeze(0).to(device)
        crop_img = crop_set_symbol(rgb, crop_cfg)
        crop_tensor = crop_transform(crop_img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = build_card_embedding(model, full_tensor, crop_tensor).cpu().numpy().flatten()

    search_cfg = config.get("recognition", {})
    top_k = search_cfg.get("top_k", 5)
    threshold = search_cfg.get("threshold", 0.88)
    results = db.search_similar(query_embedding, top_k=top_k, threshold=threshold)

    for rank, hit in enumerate(results, start=1):
        print(f"[Rank {rank}] {hit['name']} - cos={hit['similarity']:.3f}")

    elapsed = time.time() - start_time
    if results:
        best_match = results[0]
        return best_match, best_match["similarity"], elapsed
    return None, 0.0, elapsed


def test_all_camera_images(
    config: dict,
    model_path: str,
    camera_dir: str,
    output_dir: str,
    device: torch.device,
    config_path: str,
) -> None:
    """Iteriert ueber alle Kamera-Bilder, erstellt Vergleichsplots und Statistiken."""
    print(f"[LOAD] Lade trainiertes Modell: {model_path}")
    embed_dim = config["model"]["embed_dim"]
    model = load_encoder(model_path, embed_dim=embed_dim, device=device)
    model.eval()
    print(f"[INFO] Erkennung verwendet Modell: {model_path}")

    print("[INFO] Lade Embedding-Datenbank ...")
    db_path = config.get("database", {}).get("path", "./data/cards.json")
    db = SimpleCardDB(db_path=db_path, config_path=config_path)
    print(f"[INFO] Erkennung verwendet Embedding-DB: {db.db_path}")
    if len(db.cards) == 0:
        print("[WARN] Database ist leer. Bitte zuerst generate_embeddings.py ausfuehren.")
        return
    print(f"[INFO] Database geladen: {len(db.cards)} Karten")
    db_meta = getattr(db, "meta", {})
    expected_model = os.path.abspath(model_path)
    db_model = db_meta.get("model_path")
    if db_model and os.path.abspath(db_model) != expected_model:
        print(
            "[WARN] Embedding-DB wurde mit einem anderen Modell erzeugt: "
            f"{db_model}. Bitte generate_embeddings.py mit dem aktuellen Modell neu ausfuehren."
        )

    resize_hw = resolve_resize_hw(config, config["data"]["scryfall_images"])
    target_height, target_width = resize_hw
    transform = get_inference_transforms(resize_hw)
    print(f"[INFO] Inference-Bildgroesse (Breite x Hoehe): {target_width}x{target_height}")

    crop_cfg = get_set_symbol_crop_cfg(config)
    crop_height = crop_cfg.get("target_height", 64) if crop_cfg else 64
    crop_width = crop_cfg.get("target_width", 160) if crop_cfg else 160
    crop_transform = build_resize_normalize_transform((crop_height, crop_width))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    camera_path = Path(camera_dir)
    camera_files: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        camera_files.extend(camera_path.glob(ext))
    camera_files.sort()

    if not camera_files:
        print(f"[WARN] Keine Camera-Bilder gefunden in: {camera_dir}")
        return

    print(f"[INFO] Gefunden: {len(camera_files)} Camera-Bilder")
    print(f"[INFO] Output-Verzeichnis: {output_dir}")

    total_time = 0.0
    match_scores: List[float] = []
    total_searches = len(camera_files)

    print("\n[RUN] Starte Similarity-Search fuer alle Camera-Bilder ...")
    for idx, camera_file in enumerate(tqdm(camera_files, desc="Camera-Bilder", unit="img"), start=1):
        try:
            print(f"\n[{idx}/{total_searches}] [RUN] Verarbeite: {camera_file.name}")
            best_match, similarity_score, search_time = search_camera_image(
                model,
                transform,
                crop_transform,
                db,
                str(camera_file),
                device,
                config,
                crop_cfg,
            )
            total_time += search_time
            match_scores.append(float(similarity_score))

            if best_match:
                print(f"   [MATCH] {best_match['name']} (cos={similarity_score:.4f})")
                print(f"   [TIME]  {search_time*1000:.2f} ms")
                with Image.open(camera_file) as cam_img_raw:
                    camera_img = cam_img_raw.convert("RGB")
                    comparison_path = output_path / f"{camera_file.stem}_comparison.png"
                    create_comparison_plot(
                        str(camera_file),
                        camera_img,
                        best_match,
                        similarity_score,
                        search_time,
                        str(comparison_path),
                    )
            else:
                print("   [WARN] Kein Match gefunden.")
        except Exception as exc:
            print(f"   [ERROR] Fehler bei {camera_file.name}: {exc}")

    if not match_scores:
        print("[WARN] Keine Similarity-Werte gesammelt.")
        return

    print("\n" + "=" * 60)
    print("[SUMMARY] SIMILARITY SEARCH")
    print("=" * 60)
    print(f"[INFO] Durchsuchte Bilder: {total_searches}")
    print(f"[INFO] Gesamtzeit: {total_time:.2f}s")
    avg_time_ms = (total_time / max(total_searches, 1)) * 1000
    print(f"[INFO] Durchschnitt pro Bild: {avg_time_ms:.2f} ms")
    print(f"[INFO] Beste Similarity: {max(match_scores):.4f}")
    print(f"[INFO] Schlechteste Similarity: {min(match_scores):.4f}")
    print(f"[INFO] Durchschnittliche Similarity: {np.mean(match_scores):.4f}")
    print(f"[INFO] Vergleichsgrafiken gespeichert in: {output_dir}")

    summary_path = output_path / "summary_statistics.png"
    create_summary_plot(match_scores, total_time, total_searches, str(summary_path))


def create_summary_plot(
    match_scores: Sequence[float],
    total_time: float,
    total_searches: int,
    output_path: str,
) -> None:
    """Erstellt Histogramm, Boxplot und weitere Kennzahlen als Uebersicht."""
    if not match_scores:
        print("[WARN] Keine Daten fuer Statistik-Plot vorhanden.")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    ax1.hist(match_scores, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    ax1.set_title("Similarity Score Verteilung", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Cosine Similarity")
    ax1.set_ylabel("Anzahl Bilder")
    ax1.axvline(
        np.mean(match_scores),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Durchschnitt: {np.mean(match_scores):.3f}",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.boxplot(match_scores, vert=True, patch_artist=True, boxprops=dict(facecolor="lightgreen", alpha=0.7))
    ax2.set_title("Similarity Score Box Plot", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Cosine Similarity")
    ax2.grid(True, alpha=0.3)

    avg_time_ms = (total_time / max(total_searches, 1)) * 1000
    performance_labels = ["Durchschnitt\nZeit/Bild", "Gesamt-\nzeit (ms)"]
    performance_values = [avg_time_ms, total_time * 1000]
    bars = ax3.bar(performance_labels, performance_values, color=["orange", "red"], alpha=0.7, edgecolor="black")
    ax3.set_title("Performance-Metriken", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Zeit (ms)")
    ax3.grid(True, alpha=0.3)
    for bar, value in zip(bars, performance_values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(performance_values) * 0.01,
            f"{value:.1f} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    excellent = sum(1 for score in match_scores if score > 0.8)
    good = sum(1 for score in match_scores if 0.6 < score <= 0.8)
    poor = sum(1 for score in match_scores if 0.4 < score <= 0.6)
    no_match = sum(1 for score in match_scores if score <= 0.4)
    categories = ["Excellent\n(>0.8)", "Good\n(0.6-0.8)", "Poor\n(0.4-0.6)", "No Match\n(<=0.4)"]
    counts = [excellent, good, poor, no_match]
    colors = ["green", "orange", "red", "darkred"]
    ax4.pie(counts, labels=categories, colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax4.set_title("Match Quality Verteilung", fontsize=14, fontweight="bold")

    fig.suptitle(
        f"MTG Card Recognition - Similarity Search Statistiken ({total_searches} Bilder)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[PLOT] Statistik-Grafik gespeichert: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MTG Card Recognition - Erkennung von Karten aus Pi Camera-Bildern"
    )
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Pfad zur Config-Datei.")
    parser.add_argument("--camera-dir", type=str, default=None, help="Override fuer das Kamera-Verzeichnis.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override fuer das Output-Verzeichnis.")
    parser.add_argument("--model-path", type=str, default=None, help="Override fuer das Encoder-Gewicht.")
    return parser.parse_args()


def main() -> None:
    play_button_mode = len(sys.argv) == 1
    args = parse_args()
    if play_button_mode:
        print("[INFO] Play-Button Modus: nutze Parameter aus config.yaml")

    try:
        config = load_config(args.config)
        model_path = args.model_path or config["model"]["weights_path"]
        camera_dir = args.camera_dir or config["data"]["camera_images"]
        output_dir = args.output_dir or config["data"]["output_dir"]

        device = torch.device("cuda" if torch.cuda.is_available() and config["hardware"]["use_cuda"] else "cpu")
        print("[INFO] MTG Card Similarity Search")
        print(f"[INFO] Device: {device.type.upper()}")
        if device.type == "cuda":
            print(f"       GPU: {torch.cuda.get_device_name()}")
        print(f"[INFO] Config: {args.config}")
        print(f"[INFO] Modell: {model_path}")
        print(f"[INFO] Camera-Bilder: {camera_dir}")
        print(f"[INFO] Output: {output_dir}")

        if not os.path.exists(model_path):
            print(f"[ERROR] Modell nicht gefunden: {model_path}")
            print("        Bitte train_triplet.py ausfuehren.")
            return

        camera_path = Path(camera_dir)
        if not camera_path.exists():
            print(f"[ERROR] Camera-Verzeichnis nicht gefunden: {camera_dir}")
            print("        Erstelle Verzeichnis und kopiere Testbilder hinein.")
            camera_path.mkdir(parents=True, exist_ok=True)
            return

        db_path = config["database"]["path"]
        if not os.path.exists(db_path):
            print(f"[ERROR] Database nicht gefunden: {db_path}")
            print("        Bitte zuerst Embeddings generieren (generate_embeddings.py).")
            return

        preview_files: List[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            preview_files.extend(camera_path.glob(ext))
        if not preview_files:
            print(f"[ERROR] Keine Bilder im Camera-Verzeichnis gefunden: {camera_dir}")
            print("        Kopiere einige Testbilder ins Camera-Verzeichnis.")
            return

        test_all_camera_images(config, model_path, camera_dir, output_dir, device, args.config)
    except Exception as exc:
        print(f"[ERROR] Fehler: {exc}")
        print("\n[INFO] Verwendung:")
        print("   Direkt: python src/cardscanner/recognize_cards.py")
        print("   Mit Parametern: python src/cardscanner/recognize_cards.py --camera-dir data/camera_images")
        return


if __name__ == "__main__":
    main()
