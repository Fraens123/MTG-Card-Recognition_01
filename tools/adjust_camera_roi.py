#!/usr/bin/env python3
"""
Interactive Camera ROI Adjuster

Interaktives Tool zum Einstellen der camera.card_roi Parameter.
Zeigt Pi-Cam-Bilder mit ROI-Overlay und ermöglicht Live-Anpassung.

Usage:
    python tools/adjust_camera_roi.py --config config.train20k.yaml
    python tools/adjust_camera_roi.py --config config.yaml --image data/camera_images/test.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
import numpy as np
from PIL import Image
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ROIAdjuster:
    def __init__(self, config_path: str, image_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self.load_config()
        
        # Lade ROI-Parameter aus Config oder verwende Defaults
        camera_cfg = self.config.get("camera", {})
        roi_cfg = camera_cfg.get("card_roi", {})
        self.x_min = float(roi_cfg.get("x_min", 0.0))
        self.y_min = float(roi_cfg.get("y_min", 0.0))
        self.x_max = float(roi_cfg.get("x_max", 1.0))
        self.y_max = float(roi_cfg.get("y_max", 1.0))
        
        # Lade Test-Bild
        if image_path:
            self.image_path = Path(image_path)
        else:
            # Verwende erstes Bild aus camera_dir
            paths_cfg = self.config.get("paths", {})
            camera_dir = Path(paths_cfg.get("camera_dir", "./data/camera_images"))
            images = list(camera_dir.glob("*.jpg")) + list(camera_dir.glob("*.png"))
            if not images:
                raise FileNotFoundError(f"Keine Bilder in {camera_dir}")
            self.image_path = images[0]
        
        print(f"[INFO] Lade Bild: {self.image_path}")
        self.img = Image.open(self.image_path).convert("RGB")
        self.img_width, self.img_height = self.img.size
        
        # UI Setup
        self.fig, (self.ax_main, self.ax_crop) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.subplots_adjust(left=0.05, bottom=0.25, right=0.95, top=0.95, wspace=0.2)
        
        # Slider
        slider_color = 'lightgoldenrodyellow'
        ax_x_min = plt.axes([0.15, 0.15, 0.35, 0.03], facecolor=slider_color)
        ax_x_max = plt.axes([0.15, 0.10, 0.35, 0.03], facecolor=slider_color)
        ax_y_min = plt.axes([0.15, 0.05, 0.35, 0.03], facecolor=slider_color)
        ax_y_max = plt.axes([0.15, 0.00, 0.35, 0.03], facecolor=slider_color)
        
        self.slider_x_min = Slider(ax_x_min, 'X Min', 0.0, 1.0, valinit=self.x_min, valstep=0.01)
        self.slider_x_max = Slider(ax_x_max, 'X Max', 0.0, 1.0, valinit=self.x_max, valstep=0.01)
        self.slider_y_min = Slider(ax_y_min, 'Y Min', 0.0, 1.0, valinit=self.y_min, valstep=0.01)
        self.slider_y_max = Slider(ax_y_max, 'Y Max', 0.0, 1.0, valinit=self.y_max, valstep=0.01)
        
        # Buttons
        ax_save = plt.axes([0.60, 0.05, 0.15, 0.05])
        ax_reset = plt.axes([0.60, 0.10, 0.15, 0.05])
        self.btn_save = Button(ax_save, 'Save to Config', color='lightgreen')
        self.btn_reset = Button(ax_reset, 'Reset (1:1)', color='lightcoral')
        
        # Event Handler
        self.slider_x_min.on_changed(self.update)
        self.slider_x_max.on_changed(self.update)
        self.slider_y_min.on_changed(self.update)
        self.slider_y_max.on_changed(self.update)
        self.btn_save.on_clicked(self.save_config)
        self.btn_reset.on_clicked(self.reset)
        
        # Initiales Rendering
        self.rect = None
        self.update(None)
        plt.show()
    
    def load_config(self) -> dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    def update(self, val):
        """Update ROI-Overlay und Crop-Preview"""
        self.x_min = self.slider_x_min.val
        self.x_max = self.slider_x_max.val
        self.y_min = self.slider_y_min.val
        self.y_max = self.slider_y_max.val
        
        # Validierung
        if self.x_max <= self.x_min:
            self.x_max = min(1.0, self.x_min + 0.1)
            self.slider_x_max.set_val(self.x_max)
        if self.y_max <= self.y_min:
            self.y_max = min(1.0, self.y_min + 0.1)
            self.slider_y_max.set_val(self.y_max)
        
        # Berechne Pixel-Koordinaten
        x0 = int(self.x_min * self.img_width)
        y0 = int(self.y_min * self.img_height)
        x1 = int(self.x_max * self.img_width)
        y1 = int(self.y_max * self.img_height)
        
        # Main View: Original + ROI-Overlay
        self.ax_main.clear()
        self.ax_main.imshow(self.img)
        self.ax_main.set_title("Original Image with ROI Overlay", fontsize=14, fontweight="bold")
        self.ax_main.axis("off")
        
        # ROI-Rechteck
        width = x1 - x0
        height = y1 - y0
        rect = patches.Rectangle(
            (x0, y0), width, height,
            linewidth=3, edgecolor='lime', facecolor='none', linestyle='--'
        )
        self.ax_main.add_patch(rect)
        
        # Info-Text
        info = f"ROI: x=[{self.x_min:.2f}, {self.x_max:.2f}], y=[{self.y_min:.2f}, {self.y_max:.2f}]\n"
        info += f"Pixel: ({x0}, {y0}) -> ({x1}, {y1})\n"
        info += f"Size: {width}x{height}px"
        self.ax_main.text(
            0.02, 0.98, info,
            transform=self.ax_main.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            family='monospace'
        )
        
        # Crop View: Ausgeschnittener Bereich
        self.ax_crop.clear()
        cropped = self.img.crop((x0, y0, x1, y1))
        self.ax_crop.imshow(cropped)
        self.ax_crop.set_title("Cropped Card Region", fontsize=14, fontweight="bold")
        self.ax_crop.axis("off")
        
        self.fig.canvas.draw_idle()
    
    def reset(self, event):
        """Reset auf Full-Image (0.0 - 1.0)"""
        self.slider_x_min.set_val(0.0)
        self.slider_x_max.set_val(1.0)
        self.slider_y_min.set_val(0.0)
        self.slider_y_max.set_val(1.0)
    
    def save_config(self, event):
        """Speichere ROI-Parameter in Config-Datei"""
        # Config neu laden (falls extern geändert)
        self.config = self.load_config()
        
        # Update camera.card_roi
        if "camera" not in self.config:
            self.config["camera"] = {}
        if "card_roi" not in self.config["camera"]:
            self.config["camera"]["card_roi"] = {}
        
        self.config["camera"]["card_roi"]["x_min"] = float(self.x_min)
        self.config["camera"]["card_roi"]["y_min"] = float(self.y_min)
        self.config["camera"]["card_roi"]["x_max"] = float(self.x_max)
        self.config["camera"]["card_roi"]["y_max"] = float(self.y_max)
        
        # Backup erstellen
        backup_path = Path(self.config_path).with_suffix('.yaml.bak')
        import shutil
        shutil.copy2(self.config_path, backup_path)
        print(f"[BACKUP] Backup erstellt: {backup_path}")
        
        # Speichern
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"[SAVED] ROI-Parameter gespeichert in: {self.config_path}")
        print(f"  x_min: {self.x_min:.3f}")
        print(f"  y_min: {self.y_min:.3f}")
        print(f"  x_max: {self.x_max:.3f}")
        print(f"  y_max: {self.y_max:.3f}")
        
        # Visuelles Feedback
        self.btn_save.color = 'green'
        self.btn_save.label.set_text('✓ Saved!')
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Camera ROI Adjuster - Stelle card_roi Parameter visuell ein"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Pfad zur Config-Datei (default: config.yaml)"
    )
    parser.add_argument(
        "--image", "-i",
        default=None,
        help="Spezifisches Test-Bild (default: erstes Bild aus camera_dir)"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config nicht gefunden: {config_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Camera ROI Adjuster")
    print("=" * 60)
    print(f"Config: {config_path}")
    print("\nAnleitung:")
    print("  1. Verwende die Slider um ROI-Bereich anzupassen")
    print("  2. Linkes Bild: Original mit ROI-Overlay")
    print("  3. Rechtes Bild: Preview des Crops")
    print("  4. 'Save to Config' speichert Parameter")
    print("  5. 'Reset' setzt auf Full-Image zurück")
    print("=" * 60)
    
    try:
        adjuster = ROIAdjuster(str(config_path), args.image)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
