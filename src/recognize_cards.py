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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.core.model_builder import load_encoder
from src.core.card_database import SimpleCardDB
from src.core.image_ops import (
    build_resize_normalize_transform,
    crop_card_art,
    get_full_art_crop_cfg,
    resolve_resize_hw,
)
from src.core.embedding_utils import build_card_embedding
from src.core.recognition_engine import (
    EmbeddingIndex,
    recognize_card,
    debug_print_recognition_result,
    PREFERRED_LANGS,
)


def crop_card_roi(img: Image.Image, roi_cfg: Optional[Dict]) -> Image.Image:
    """
    Schneidet die MTG-Karte aus dem Pi-Cam-Bild anhand fester relativer Koordinaten.
    roi_cfg: Dict mit x_min, y_min, x_max, y_max in [0.0, 1.0].
    Wenn roi_cfg None ist oder unvollständig, wird das Bild unverändert zurückgegeben.
    """
    if not roi_cfg:
        return img

    w, h = img.size
    try:
        x_min = float(roi_cfg.get("x_min", 0.0))
        y_min = float(roi_cfg.get("y_min", 0.0))
        x_max = float(roi_cfg.get("x_max", 1.0))
        y_max = float(roi_cfg.get("y_max", 1.0))
    except Exception:
        return img

    x0 = int(x_min * w)
    y0 = int(y_min * h)
    x1 = int(x_max * w)
    y1 = int(y_max * h)

    # Bounds sichern
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))

    return img.crop((x0, y0, x1, y1))


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


class CropDebugger:
    def __init__(self, directory: Optional[Path], limit: int):
        self.directory = directory
        self.limit = max(0, int(limit or 0))
        self.count = 0
        if self.limit > 0 and self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def record(self, card_img: Optional[Image.Image], art_img: Image.Image, source: str):
        if self.limit <= 0 or self.directory is None or self.count >= self.limit:
            return
        stem = Path(source).stem
        art_path = self.directory / f"{stem}_art_{self.count:02d}.png"
        card_path = self.directory / f"{stem}_card_{self.count:02d}.png" if card_img else None
        try:
            art_img.save(art_path)
            if card_img and card_path:
                card_img.save(card_path)
            self.count += 1
        except Exception as exc:
            print(f"[WARN] Konnte Kamera-Crop nicht speichern ({source}): {exc}")


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
    index: EmbeddingIndex,
    camera_img_path: str,
    device: torch.device,
    config: dict,
    art_crop_cfg: Optional[dict],
    debug_recorder=None,
) -> Tuple[Optional[Dict], float, float]:
    """
    Erkennt eine Karte aus einem Kamerabild mit der neuen Recognition-Engine.
    
    Datenfluss:
    1. Bild laden und ROI schneiden
    2. recognize_card() aufrufen (Bild → Embedding → Scryfall-ID → Oracle-ID)
    3. Ergebnis in altes Format konvertieren (für Kompatibilität)
    """
    start_time = time.time()

    with Image.open(camera_img_path) as img:
        rgb = img.convert("RGB")
        # Karte per ROI aus Pi-Cam-Bild schneiden
        camera_cfg = config.get("camera", {})
        card_roi_cfg = camera_cfg.get("card_roi", None)
        card_img = crop_card_roi(rgb, card_roi_cfg)

        if debug_recorder:
            art_img = crop_card_art(card_img, art_crop_cfg)
            debug_recorder(card_img, art_img, camera_img_path)
        
        # Neue Recognition-Engine verwenden
        # Datenfluss: Bild → CNN → Query-Embedding → Top-k Prints (Scryfall-ID) → Oracle-Gruppierung
        result = recognize_card(
            image=card_img,
            model=model,
            index=index,
            transform=transform,
            crop_cfg=art_crop_cfg,
            device=device,
            k=20,  # Top-20 Prints
            preferred_langs=PREFERRED_LANGS,  # ["de", "en"]
        )

    elapsed = time.time() - start_time
    
    if result is None:
        return None, 0.0, elapsed
    
    # Debug-Output
    debug_print_recognition_result(result)
    
    # Konvertiere zu altem Format (für create_comparison_plot)
    best_match = {
        "card_uuid": result.scryfall_id,  # Scryfall-ID (konkreter Print)
        "oracle_id": result.oracle_id,     # Oracle-ID (logische Karte)
        "name": result.meta.name,
        "set_code": result.meta.set_code,
        "collector_number": result.meta.collector_number,
        "image_path": result.meta.image_path,
        "lang": result.meta.lang,
        "similarity": result.similarity,
    }
    
    return best_match, result.similarity, elapsed


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
    model = load_encoder(model_path, cfg=config, device=device)
    model.eval()
    print(f"[INFO] Erkennung verwendet Modell: {model_path}")

    print("[INFO] Lade Embedding-Index (neue Recognition-Engine) ...")
    db_path = config.get("database", {}).get("sqlite_path", "tcg_database/database/karten.db")
    emb_dim = config.get("encoder", {}).get("emb_dim", config.get("model", {}).get("embed_dim", 1024))
    
    # Neue EmbeddingIndex-Klasse verwenden (gruppiert nach Scryfall-ID!)
    index = EmbeddingIndex(db_path=db_path, mode="runtime", emb_dim=emb_dim)
    
    print(f"[INFO] Embedding-Index geladen: {index.size()} Embeddings")
    if index.size() == 0:
        print("[WARN] Index ist leer. Bitte zuerst export_embeddings.py ausfuehren.")
        return

    paths_cfg = config.get("paths", {})
    resize_hw = resolve_resize_hw(config, paths_cfg.get("scryfall_dir"))
    target_height, target_width = resize_hw
    transform = get_inference_transforms(resize_hw)
    print(f"[INFO] Inference-Bildgroesse (Breite x Hoehe): {target_width}x{target_height}")

    art_cfg = get_full_art_crop_cfg(config)

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

    print("\n[RUN] Starte Recognition mit neuer Engine (Scryfall-ID → Oracle-ID) ...")

    for idx, camera_file in enumerate(tqdm(camera_files, desc="Camera-Bilder", unit="img"), start=1):
        try:
            print(f"\n[{idx}/{total_searches}] [RUN] Verarbeite: {camera_file.name}")
            best_match, similarity_score, search_time = search_camera_image(
                model,
                transform,
                index,
                str(camera_file),
                device,
                config,
                art_cfg,
                None,
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
            import traceback
            traceback.print_exc()

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
        paths_cfg = config.get("paths", {})
        default_model_path = os.path.join(paths_cfg.get("models_dir", "./models"), "encoder_fine.pt")
        model_path = args.model_path or default_model_path
        camera_dir = args.camera_dir or paths_cfg.get("camera_dir", "./data/camera_images")
        output_dir = args.output_dir or paths_cfg.get("output_dir", "./output_matches")

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
            print("        Bitte zuerst train_coarse.py und train_fine.py ausfuehren.")
            return

        camera_path = Path(camera_dir)
        if not camera_path.exists():
            print(f"[ERROR] Camera-Verzeichnis nicht gefunden: {camera_dir}")
            print("        Erstelle Verzeichnis und kopiere Testbilder hinein.")
            camera_path.mkdir(parents=True, exist_ok=True)
            return

        db_path = config.get("database", {}).get("sqlite_path", "tcg_database/database/karten.db")
        if not os.path.exists(db_path):
            print(f"[ERROR] Database nicht gefunden: {db_path}")
            print("        Bitte zuerst Embeddings generieren (export_embeddings.py).")
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
