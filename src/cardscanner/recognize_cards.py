#!/usr/bin/env python3
"""
MTG Card Recognition Script
Erkennt MTG-Karten aus Pi Camera-Bildern und vergleicht sie mit Scryfall-Referenzen.
Nutzt Cosine-Similarity für die Kartenerkennung und erstellt Side-by-Side Vergleichsgrafiken.
"""

import os
import sys
from PIL import Image
import time
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm

# Absolute Imports für direkten Aufruf
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Try relative imports first, fallback to absolute
try:
    from .model import load_encoder
    from .db import SimpleCardDB
    from .dataset import parse_scryfall_filename
except ImportError:
    # Fallback für direkten Aufruf
    from src.cardscanner.model import load_encoder
    from src.cardscanner.db import SimpleCardDB
    from src.cardscanner.dataset import parse_scryfall_filename

try:
    from .crop_utils import crop_set_symbol
except ImportError:
    from src.cardscanner.crop_utils import crop_set_symbol


def load_config(config_path: str = "config.yaml") -> dict:
    """Lädt YAML-Konfiguration direkt"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config-Datei nicht gefunden: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise ValueError(f"Config-Datei ist leer: {config_path}")
    
    return config


def get_inference_transforms(resize_hw: tuple = (320, 224)) -> T.Compose:
    """Transform-Pipeline für Inference (identisch zur Embedding-Generierung). resize_hw = (H, W)."""
    return T.Compose([
        T.Resize(resize_hw, antialias=True),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def create_comparison_plot(camera_img_path: str, camera_img: Image.Image, 
                          best_match: Dict, similarity_score: float,
                          search_time: float, output_path: str):
    """
    Erstellt eine Side-by-Side Vergleichsgrafik zwischen Camera- und Scryfall-Bild
    """
    # Setup matplotlib mit größerem Plot und Unicode-fähiger Schriftart
    import matplotlib
    # Versuche Noto Sans Symbols als Schriftart zu setzen (falls installiert)
    try:
        matplotlib.rcParams['font.family'] = 'Noto Sans Symbols'
    except Exception:
        pass
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.2, 
                  height_ratios=[0.1, 0.8, 0.1], width_ratios=[1, 1])
    
    # Titel-Zeile
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')
    title_ax.text(0.5, 0.5, f"MTG Card Recognition - Similarity Search Results", 
                  fontsize=20, fontweight='bold', ha='center', va='center')
    
    # Camera-Bild (links)
    camera_ax = fig.add_subplot(gs[1, 0])
    camera_ax.imshow(camera_img)
    camera_ax.set_title("📷 Pi Camera Image", fontsize=16, fontweight='bold', pad=20)
    camera_ax.axis('off')
    
    # Scryfall-Bild (rechts) 
    scryfall_ax = fig.add_subplot(gs[1, 1])
    if os.path.exists(best_match['image_path']):
        scryfall_img = Image.open(best_match['image_path']).convert('RGB')
        scryfall_ax.imshow(scryfall_img)
    else:
        # Fallback wenn Bild nicht gefunden
        scryfall_ax.text(0.5, 0.5, "Bild nicht gefunden", ha='center', va='center', 
                        fontsize=14, transform=scryfall_ax.transAxes)
    
    scryfall_ax.set_title("🃏 Best Match (Scryfall)", fontsize=16, fontweight='bold', pad=20)
    scryfall_ax.axis('off')
    
    # Info-Bereich (unten)
    info_ax = fig.add_subplot(gs[2, :])
    info_ax.axis('off')
    
    # Informationstext erstellen
    camera_filename = Path(camera_img_path).name
    info_text = f"""
📊 SEARCH RESULTS:
• Camera File: {camera_filename}
• Best Match: {best_match['name']} (Set: {best_match['set_code']}, #{best_match['collector_number']})
• Cosine Similarity: {similarity_score:.4f} ({similarity_score*100:.1f}%)
• Search Time: {search_time*1000:.2f}ms
• Card UUID: {best_match['card_uuid'][:8]}...
• Image Path: {Path(best_match['image_path']).name}
    """.strip()
    
    # Farbkodierung basierend auf Similarity-Score
    if similarity_score > 0.8:
        color = 'green'
        status = "🎯 EXCELLENT MATCH"
    elif similarity_score > 0.6:
        color = 'orange' 
        status = "⚠️ GOOD MATCH"
    elif similarity_score > 0.4:
        color = 'red'
        status = "❌ POOR MATCH"
    else:
        color = 'darkred'
        status = "❌ NO MATCH"
    
    info_ax.text(0.02, 0.95, info_text, fontsize=12, ha='left', va='top', 
                 transform=info_ax.transAxes, family='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    info_ax.text(0.98, 0.95, status, fontsize=14, fontweight='bold', 
                 ha='right', va='top', transform=info_ax.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8, edgecolor='black'))
    
    # Similarity-Score Balken
    bar_width = 0.6
    bar_x = 0.2
    bar_y = 0.2
    
    # Hintergrund-Balken
    info_ax.add_patch(patches.Rectangle((bar_x, bar_y), bar_width, 0.15, 
                                       facecolor='lightgray', edgecolor='black', linewidth=2))
    # Similarity-Balken
    info_ax.add_patch(patches.Rectangle((bar_x, bar_y), bar_width * similarity_score, 0.15,
                                       facecolor=color, alpha=0.8, edgecolor='black', linewidth=2))
    
    info_ax.text(bar_x + bar_width/2, bar_y + 0.075, f"{similarity_score:.3f}", 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Speichern
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   💾 Vergleichsgrafik gespeichert: {output_path}")


def search_camera_image(model: torch.nn.Module, transform: T.Compose, 
                    db: SimpleCardDB, camera_img_path: str, 
                    device: torch.device, config: dict) -> Tuple[Dict, float, float]:
    """
    Führt Similarity-Search für ein Camera-Bild durch
    """
    start_time = time.time()
    

    # Bild laden und transformieren (Fullbild)
    with Image.open(camera_img_path) as img:
        img_rgb = img.convert('RGB')
        img_tensor = transform(img_rgb).unsqueeze(0).to(device)
        # Set-Symbol-Crop
        crop_cfg = config.get('debug', {}).get('set_symbol_crop', None)
        crop_img = crop_set_symbol(img_rgb, crop_cfg)
        crop_tensor = transform(crop_img).unsqueeze(0).to(device)

    # Embedding generieren (Full + Crop)
    with torch.no_grad():
        emb_full = model(img_tensor)
        emb_crop = model(crop_tensor)
        emb_full = nn.functional.normalize(emb_full, p=2, dim=-1)
        emb_crop = nn.functional.normalize(emb_crop, p=2, dim=-1)
        query_embedding = torch.cat([emb_full, emb_crop], dim=-1).cpu().numpy().flatten()

    # Suche in Database
    top_k = config.get('recognition', {}).get('top_k', 5)
    threshold = config.get('recognition', {}).get('threshold', 0.88)
    results = db.search_similar(query_embedding, top_k=top_k, threshold=threshold)

    for rank, hit in enumerate(results, start=1):
        print(f"[Rank {rank}] {hit['name']} – cos={hit['similarity']:.3f}")

    search_time = time.time() - start_time

    if results:
        best_match = results[0]
        similarity_score = best_match['similarity']
        return best_match, similarity_score, search_time
    else:
        return None, 0.0, search_time


def test_all_camera_images(config: dict, model_path: str, camera_dir: str, 
                          output_dir: str, device: torch.device):
    """
    Testet alle Camera-Bilder und erstellt Vergleichsgrafiken
    """
    print(f"🔧 Lade trainiertes Modell: {model_path}")
    embed_dim = config['model']['embed_dim']
    model = load_encoder(model_path, embed_dim=embed_dim, device=device)
    model.eval()
    
    print(f"📂 Lade Database...")
    db = SimpleCardDB()
    
    if len(db.cards) == 0:
        print("❌ Database ist leer! Bitte zuerst Embeddings generieren.")
        return
    
    print(f"🎯 Database geladen: {len(db.cards)} Karten")
    
    # Transform für Inference
    if config['training']['auto_detect_size']:
        try:
            from .train_triplet import detect_image_size
        except ImportError:
            from src.cardscanner.train_triplet import detect_image_size
        target_width, target_height = detect_image_size(camera_dir)
        target_size = (target_height, target_width)
    else:
        target_width = config['training']['target_width']
        target_height = config['training']['target_height']
        target_size = (target_height, target_width)
    
    transform = get_inference_transforms(target_size)
    print(f"📐 Bildgröße (Breite x Höhe): {target_width}x{target_height}")
    
    # Output-Verzeichnis erstellen
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Alle Camera-Bilder finden
    camera_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        camera_files.extend(Path(camera_dir).glob(ext))
    
    if not camera_files:
        print(f"❌ Keine Camera-Bilder gefunden in: {camera_dir}")
        return
    
    print(f"📸 Gefunden: {len(camera_files)} Camera-Bilder")
    print(f"📁 Output-Verzeichnis: {output_dir}")
    
    # Statistiken
    total_searches = len(camera_files)
    total_time = 0
    match_scores = []
    
    print(f"\n🔍 Starte Similarity-Search für alle Camera-Bilder...")
    
    # Progress Bar für alle Camera-Bilder
    for i, camera_file in enumerate(tqdm(camera_files, desc="Teste Camera-Bilder", unit="img"), 1):
        try:
            print(f"\n[{i}/{total_searches}] 🔍 Verarbeite: {camera_file.name}")
            
            # Similarity-Search durchführen
            best_match, similarity_score, search_time = search_camera_image(
                model, transform, db, str(camera_file), device, config
            )
            
            total_time += search_time
            match_scores.append(similarity_score)
            
            if best_match:
                print(f"   🎯 Best Match: {best_match['name']} ({similarity_score:.4f})")
                print(f"   ⏱️ Search Time: {search_time*1000:.2f}ms")
                
                # Vergleichsgrafik erstellen
                camera_img = Image.open(camera_file).convert('RGB')
                output_filename = f"{camera_file.stem}_comparison.png"
                output_filepath = output_path / output_filename
                
                create_comparison_plot(
                    str(camera_file), camera_img, best_match, 
                    similarity_score, search_time, str(output_filepath)
                )
            else:
                print(f"   ❌ Kein Match gefunden!")
        
        except Exception as e:
            print(f"   ❌ Fehler bei {camera_file.name}: {e}")
            continue
    
    # Zusammenfassung
    print(f"\n" + "="*60)
    print(f"📊 SIMILARITY SEARCH ZUSAMMENFASSUNG")
    print(f"="*60)
    print(f"🔍 Durchsuchte Bilder: {total_searches}")
    print(f"⏱️ Gesamtzeit: {total_time:.2f}s")
    print(f"🚀 Durchschnitt pro Bild: {(total_time/total_searches)*1000:.2f}ms")
    print(f"📈 Beste Similarity: {max(match_scores):.4f}")
    print(f"📉 Schlechteste Similarity: {min(match_scores):.4f}")
    print(f"📊 Durchschnittliche Similarity: {np.mean(match_scores):.4f}")
    print(f"📁 Vergleichsgrafiken gespeichert in: {output_dir}")
    
    # Erstelle Gesamt-Statistik Grafik
    create_summary_plot(match_scores, total_time, total_searches, str(output_path / "summary_statistics.png"))


def create_summary_plot(match_scores: List[float], total_time: float, 
                       total_searches: int, output_path: str):
    """
    Erstellt eine Zusammenfassungs-Grafik mit Statistiken
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Similarity Score Histogramm
    ax1.hist(match_scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_title('📊 Similarity Score Verteilung', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Anzahl Bilder')
    ax1.axvline(np.mean(match_scores), color='red', linestyle='--', linewidth=2, label=f'Durchschnitt: {np.mean(match_scores):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box Plot
    ax2.boxplot(match_scores, vert=True, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen', alpha=0.7))
    ax2.set_title('📦 Similarity Score Box Plot', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cosine Similarity')
    ax2.grid(True, alpha=0.3)
    
    # Performance Metriken
    avg_time_ms = (total_time / total_searches) * 1000
    performance_data = ['Durchschnitt\nZeit/Bild', 'Gesamt-\nzeit']
    performance_values = [avg_time_ms, total_time * 1000]
    performance_colors = ['orange', 'red']
    
    bars = ax3.bar(performance_data, performance_values, color=performance_colors, alpha=0.7, edgecolor='black')
    ax3.set_title('⏱️ Performance Metriken', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Zeit (ms)')
    ax3.grid(True, alpha=0.3)
    
    # Werte auf Balken anzeigen
    for bar, value in zip(bars, performance_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(performance_values)*0.01,
                f'{value:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Match Quality Kategorien
    excellent = sum(1 for score in match_scores if score > 0.8)
    good = sum(1 for score in match_scores if 0.6 < score <= 0.8)
    poor = sum(1 for score in match_scores if 0.4 < score <= 0.6)
    no_match = sum(1 for score in match_scores if score <= 0.4)
    
    categories = ['Excellent\n(>0.8)', 'Good\n(0.6-0.8)', 'Poor\n(0.4-0.6)', 'No Match\n(≤0.4)']
    counts = [excellent, good, poor, no_match]
    colors = ['green', 'orange', 'red', 'darkred']
    
    wedges, texts, autotexts = ax4.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 10})
    ax4.set_title('🎯 Match Quality Verteilung', fontsize=14, fontweight='bold')
    
    # Gesamt-Titel
    fig.suptitle(f'MTG Card Recognition - Similarity Search Statistiken ({total_searches} Bilder)', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📈 Statistik-Grafik gespeichert: {output_path}")


def main():
    """
    Hauptfunktion für Similarity-Search Testing
    Kann direkt ohne Parameter ausgeführt werden - nutzt dann config.yaml
    """
    parser = argparse.ArgumentParser(description="MTG Card Recognition - Erkennung von Karten aus Pi Camera-Bildern")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                       help="Pfad zur Config-Datei")
    parser.add_argument("--camera-dir", type=str, default=None,
                       help="Override: Verzeichnis mit Camera-Bildern")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Override: Ausgabeverzeichnis für Ergebnisse")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Override: Pfad zum trainierten Modell")
    
    # Falls Script ohne Argumente aufgerufen wird (z.B. Play-Button in VS Code)
    import sys
    if len(sys.argv) == 1:
        # Direkter Aufruf ohne Parameter - nutze Config-Defaults
        args = argparse.Namespace(
            config="config.yaml",
            camera_dir=None,
            output_dir=None,
            model_path=None
        )
        print("🎮 Play-Button Modus: Nutze Standard-Parameter aus config.yaml")
    else:
        args = parser.parse_args()
    
    try:
        # Config laden
        config = load_config(args.config)
        
        # Parameter aus Config oder Command-Line
        model_path = args.model_path or config['model']['weights_path']
        camera_dir = args.camera_dir or config['data']['camera_images']
        output_dir = args.output_dir or config['data']['output_dir']
        
        # Device
        device = torch.device("cuda" if torch.cuda.is_available() and config['hardware']['use_cuda'] else "cpu")
        
        print(f"🚀 MTG Card Similarity Search Testing")
        print(f"🖥️  Device: {device.type.upper()}")
        if device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"📋 Config: {args.config}")
        print(f"🔧 Modell: {model_path}")
        print(f"📷 Camera-Bilder: {camera_dir}")
        print(f"📁 Output: {output_dir}")
        
        # Prüfungen
        if not os.path.exists(model_path):
            print(f"❌ Modell nicht gefunden: {model_path}")
            print("   ➡️ Bitte zuerst Training durchführen mit train_triplet.py!")
            return
        
        if not os.path.exists(camera_dir):
            print(f"❌ Camera-Verzeichnis nicht gefunden: {camera_dir}")
            print(f"   ➡️ Erstelle Verzeichnis und kopiere Testbilder hinein!")
            os.makedirs(camera_dir, exist_ok=True)
            print(f"   📁 Verzeichnis erstellt: {camera_dir}")
            return
        
        # Database prüfen
        db_path = config['database']['path']
        if not os.path.exists(db_path):
            print(f"❌ Database nicht gefunden: {db_path}")
            print("   ➡️ Bitte zuerst Embeddings generieren mit generate_embeddings.py!")
            return
        
        # Prüfe ob Camera-Bilder vorhanden
        camera_images = list(Path(camera_dir).glob("*.jpg")) + list(Path(camera_dir).glob("*.jpeg")) + list(Path(camera_dir).glob("*.png"))
        if not camera_images:
            print(f"❌ Keine Bilder im Camera-Verzeichnis gefunden: {camera_dir}")
            print("   ➡️ Kopiere einige Testbilder ins Camera-Verzeichnis!")
            return
            
        print(f"📷 Gefunden: {len(camera_images)} Camera-Bilder")
        
        # Similarity-Search starten
        test_all_camera_images(config, model_path, camera_dir, output_dir, device)
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        print("\n💡 Verwendung:")
        print("   🎮 Direkt mit Play-Button: nutzt config.yaml Parameter")
        print("   🖥️  Mit Parametern: python src/cardscanner/recognize_cards.py --camera-dir data/camera_images")
        parser.print_help()


if __name__ == "__main__":
    main()
