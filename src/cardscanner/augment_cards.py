#!/usr/bin/env python3
"""
Script zur Augmentierung von MTG-Karten für das Training.
Simuliert Camera-Bedingungen auf Scryfall-Bildern.
"""

import os
import random
import argparse
import yaml
from pathlib import Path
import sys
import shutil
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from .crop_utils import crop_set_symbol, load_symbol_crop_cfg
except ImportError:
    from src.cardscanner.crop_utils import crop_set_symbol, load_symbol_crop_cfg


def parse_scryfall_filename(filename: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parst Scryfall-Dateinamen und liefert (card_uuid, set_code, collector_number, card_name).
    """
    name, _ = os.path.splitext(os.path.basename(filename))

    parts = name.split("_")
    if len(parts) >= 4:
        set_code = parts[0]
        collector_number = parts[1]
        card_uuid = parts[-1]
        card_name = "_".join(parts[2:-1])
        return card_uuid, set_code, collector_number, card_name

    if "-" in name:
        parts = name.split("-", 2)
        if len(parts) == 3:
            set_code = parts[0]
            collector_number = parts[1]
            card_name = parts[2]
            card_uuid = f"{set_code}_{collector_number}_{hash(name) % 100000:05d}"
            return card_uuid, set_code, collector_number, card_name

    print(f"[WARN] parse_scryfall_filename: Unbekanntes Format für '{filename}'")
    return None


def load_config(config_path: str = "config.yaml") -> dict:
    """Lädt die Konfiguration aus YAML-Datei"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️  Config-Datei nicht gefunden: {config_path}")
        return {}


class CameraLikeAugmentor:
    """Klasse für realistische Kamera-Augmentierung"""
    
    def __init__(self, **params):
        # Camera-ähnliche Parameter (alle konfigurierbar)
        self.brightness_range = params.get('brightness_range', (0.6, 1.4))  # Belichtung
        self.contrast_range = params.get('contrast_range', (0.7, 1.3))      # Kontrast
        self.saturation_range = params.get('saturation_range', (0.6, 1.2))  # Sättigung
        self.color_temperature_range = params.get('color_temperature_range', (0.8, 1.2))  # Farbtemperatur
        self.hue_shift_max = params.get('hue_shift_max', 10.0)              # Farbton-Verschiebung
        self.sharpness_range = params.get('sharpness_range', (0.5, 1.5))    # Schärfe
        self.blur_range = params.get('blur_range', (0.0, 2.0))              # Bewegungsunschärfe
        self.noise_range = params.get('noise_range', (0, 20))               # Sensor-Rauschen
        self.rotation_range = params.get('rotation_range', (-8, 8))         # Kamera-Neigung
        self.perspective_strength = params.get('perspective', 0.05)         # Perspektivische Verzerrung
        self.shadow_strength = params.get('shadow', 0.3)                    # Schatten-Simulation
        
        # Hintergrundfarbe für Rotation/Perspektive
        background_color = params.get('background_color', 'white')
        self.fill_color = (255, 255, 255) if background_color == 'white' else (0, 0, 0)
        
        # Augmentierungs-Wahrscheinlichkeiten
        self.brightness_prob = params.get('brightness_prob', 0.8)
        self.contrast_prob = params.get('contrast_prob', 0.7)
        self.saturation_prob = params.get('saturation_prob', 0.6)
        self.color_temperature_prob = params.get('color_temperature_prob', 0.5)  # NEU
        self.hue_shift_prob = params.get('hue_shift_prob', 0.4)                  # NEU
        self.sharpness_prob = params.get('sharpness_prob', 0.5)
        self.blur_prob = params.get('blur_prob', 0.4)
        self.noise_prob = params.get('noise_prob', 0.3)
        self.rotation_prob = params.get('rotation_prob', 0.6)
        self.perspective_prob = params.get('perspective_prob', 0.2)
        self.shadow_prob = params.get('shadow_prob', 0.3)
    
    def augment_exposure(self, img: Image.Image) -> Image.Image:
        """Simuliert verschiedene Belichtungen (über-/unterbelichtet)"""
        factor = random.uniform(*self.brightness_range)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def augment_contrast(self, img: Image.Image) -> Image.Image:
        """Simuliert unterschiedliche Kontraste"""
        factor = random.uniform(*self.contrast_range)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def augment_saturation(self, img: Image.Image) -> Image.Image:
        """Simuliert Farbstiche (Weißabgleich-Probleme)"""
        factor = random.uniform(*self.saturation_range)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)
    
    def augment_sharpness(self, img: Image.Image) -> Image.Image:
        """Simuliert Fokus-Probleme"""
        factor = random.uniform(*self.sharpness_range)
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)
    
    def augment_motion_blur(self, img: Image.Image) -> Image.Image:
        """Simuliert Bewegungsunschärfe"""
        radius = random.uniform(*self.blur_range)
        if radius > 0.1:
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img
    
    def augment_noise(self, img: Image.Image) -> Image.Image:
        """Simuliert Sensor-Rauschen bei schlechtem Licht"""
        intensity = random.uniform(*self.noise_range)
        if intensity > 1:
            img_array = np.array(img)
            noise = np.random.normal(0, intensity, img_array.shape)
            noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_array)
        return img
    
    def augment_rotation(self, img: Image.Image) -> Image.Image:
        """Simuliert leichte Kamera-Neigung"""
        angle = random.uniform(*self.rotation_range)
        return img.rotate(angle, expand=False, fillcolor=self.fill_color)
    
    def augment_perspective(self, img: Image.Image) -> Image.Image:
        """Simuliert leichte perspektivische Verzerrung"""
        if random.random() > self.perspective_strength:
            return img
        
        w, h = img.size
        # Leichte Trapezform simulieren
        strength = random.uniform(-self.perspective_strength, self.perspective_strength)
        offset = int(w * strength)
        
        # Perspektiv-Transform (vereinfacht)
        if abs(offset) > 2:
            # Simuliere leichte Trapezform durch horizontale Skalierung
            new_img = img.transform(
                (w, h), 
                Image.PERSPECTIVE,
                (0, offset, w, 0, w, h, 0, h - offset),
                fillcolor=self.fill_color
            )
            return new_img
        return img
    
    def augment_shadow(self, img: Image.Image) -> Image.Image:
        """Simuliert ungleichmäßige Beleuchtung/Schatten"""
        if random.random() > self.shadow_strength:
            return img
            
        # Erstelle Schatten-Maske
        w, h = img.size
        shadow_mask = Image.new('L', (w, h), 255)
        
        # Zufällige Schatten-Position
        shadow_x = random.randint(0, w//3)
        shadow_y = random.randint(0, h//3)
        shadow_w = random.randint(w//4, w//2)
        shadow_h = random.randint(h//4, h//2)
        
        # Weicher Schatten
        shadow_region = Image.new('L', (shadow_w, shadow_h), 180)
        shadow_region = shadow_region.filter(ImageFilter.GaussianBlur(radius=20))
        shadow_mask.paste(shadow_region, (shadow_x, shadow_y))
        
        # Schatten anwenden
        img_array = np.array(img)
        mask_array = np.array(shadow_mask) / 255.0
        
        for i in range(3):  # RGB channels
            img_array[:, :, i] = (img_array[:, :, i] * mask_array).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def augment_color_temperature(self, img: Image.Image) -> Image.Image:
        """Simuliert Farbtemperatur-Schwankungen der Kamera"""
        temp_factor = random.uniform(*self.color_temperature_range)
        
        # Konvertiere zu numpy für Farbkanal-Manipulation
        img_array = np.array(img).astype(np.float32)
        
        if temp_factor > 1.0:  # Wärmer (mehr rot/gelb)
            img_array[:, :, 0] *= min(temp_factor, 1.3)      # Rot verstärken
            img_array[:, :, 1] *= min(temp_factor * 0.9, 1.2) # Grün leicht verstärken
            img_array[:, :, 2] *= max(temp_factor * 0.7, 0.8) # Blau reduzieren
        else:  # Kühler (mehr blau)
            img_array[:, :, 0] *= max(temp_factor * 0.8, 0.7) # Rot reduzieren
            img_array[:, :, 1] *= max(temp_factor * 0.9, 0.8) # Grün leicht reduzieren
            img_array[:, :, 2] *= min(temp_factor * 1.2, 1.3) # Blau verstärken
        
        # Clipping und Rückkonvertierung
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def augment_hue_shift(self, img: Image.Image) -> Image.Image:
        """Simuliert Farbton-Verschiebungen (Hue-Shift)"""
        if img.mode != 'RGB':
            return img
            
        # Zufällige Hue-Verschiebung
        hue_shift = random.uniform(-self.hue_shift_max, self.hue_shift_max)
        
        # Konvertiere RGB zu HSV
        img_hsv = img.convert('HSV')
        hsv_array = np.array(img_hsv).astype(np.float32)
        
        # Verschiebe Hue-Kanal (Index 0)
        hsv_array[:, :, 0] = (hsv_array[:, :, 0] + hue_shift) % 256
        
        # Rückkonvertierung zu RGB
        img_hsv_shifted = Image.fromarray(hsv_array.astype(np.uint8), mode='HSV')
        return img_hsv_shifted.convert('RGB')
    
    def create_camera_like_augmentations(self, img: Image.Image, num_augmentations: int = 5) -> List[Image.Image]:
        """Erstellt Camera-ähnliche Augmentierungen im Original-Format"""
        augmentations = []
        
        # Original (unverändert)
        augmentations.append(img.copy())
        
        # Verschiedene Camera-Bedingungen simulieren
        for i in range(num_augmentations):
            aug_img = img.copy()  # Starte mit Original-Format!
            
            # Realistische Kamera-Effekte anwenden
            if random.random() < self.brightness_prob:
                aug_img = self.augment_exposure(aug_img)
            
            if random.random() < self.contrast_prob:
                aug_img = self.augment_contrast(aug_img)
            
            if random.random() < self.saturation_prob:
                aug_img = self.augment_saturation(aug_img)
            
            # NEU: Farbtemperatur und Hue-Shift
            if random.random() < self.color_temperature_prob:
                aug_img = self.augment_color_temperature(aug_img)
            
            if random.random() < self.hue_shift_prob:
                aug_img = self.augment_hue_shift(aug_img)
            
            if random.random() < self.sharpness_prob:
                aug_img = self.augment_sharpness(aug_img)
            
            if random.random() < self.blur_prob:
                aug_img = self.augment_motion_blur(aug_img)
            
            if random.random() < self.noise_prob:
                aug_img = self.augment_noise(aug_img)
            
            if random.random() < self.rotation_prob:
                aug_img = self.augment_rotation(aug_img)
            
            if random.random() < self.perspective_prob:
                aug_img = self.augment_perspective(aug_img)
            
            if random.random() < self.shadow_prob:
                aug_img = self.augment_shadow(aug_img)
            
            augmentations.append(aug_img)
        
        return augmentations


def main():
    """Hauptfunktion zum Augmentieren aller Scryfall-Bilder"""
    # Konfiguration laden
    config = load_config()
    aug_config = config.get('augmentation', {})
    data_cfg = config.get('data', {})
    
    parser = argparse.ArgumentParser(description="Augmentierung von MTG-Karten für Camera-ähnliche Bedingungen")
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default=data_cfg.get("scryfall_images", "./data/scryfall_images"),
        help="Verzeichnis mit Scryfall-Bildern",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=data_cfg.get("precomputed_augmented", "./data/precomputed_augmented"),
        help="Zielverzeichnis für vorberechnete Augmentierungen",
    )
    parser.add_argument(
        "--num_augmentations",
        "-n",
        type=int,
        default=aug_config.get("precompute_variants", aug_config.get("num_augmentations", 8)),
        help="Anzahl Kamera-Augmentierungen pro Karte",
    )
    parser.add_argument(
        "--jpg_quality",
        type=int,
        default=90,
        help="JPEG-Qualität (1-100) für gespeicherte Varianten",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Existierende Kartenordner im Output löschen und neu erzeugen",
    )

    default_symbol = aug_config.get("save_symbol_crops", True)
    parser.add_argument(
        "--save-symbol-crops",
        dest="save_symbol_crops",
        action="store_true",
        default=default_symbol,
        help="Set-Symbol-Crops zusätzlich speichern (Standard aus config)",
    )
    parser.add_argument(
        "--no-symbol-crops",
        dest="save_symbol_crops",
        action="store_false",
        help="Nur Vollbilder sichern, keine Symbol-Crops",
    )

    # Camera-Parameter mit Defaults aus Konfiguration
    parser.add_argument(
        "--brightness_min",
        type=float,
        default=aug_config.get("brightness_min", 0.6),
        help="Minimaler Helligkeitsfaktor (Belichtung)",
    )
    parser.add_argument(
        "--brightness_max",
        type=float,
        default=aug_config.get("brightness_max", 1.4),
        help="Maximaler Helligkeitsfaktor (Belichtung)",
    )
    parser.add_argument(
        "--contrast_min",
        type=float,
        default=aug_config.get("contrast_min", 0.7),
        help="Minimaler Kontrastfaktor",
    )
    parser.add_argument(
        "--contrast_max",
        type=float,
        default=aug_config.get("contrast_max", 1.3),
        help="Maximaler Kontrastfaktor",
    )
    parser.add_argument(
        "--blur_max",
        type=float,
        default=aug_config.get("blur_max", 2.0),
        help="Maximaler Blur-Radius (Bewegungsunschärfe)",
    )
    parser.add_argument(
        "--noise_max",
        type=float,
        default=aug_config.get("noise_max", 20.0),
        help="Maximales Sensor-Rauschen",
    )
    parser.add_argument(
        "--rotation_max",
        type=float,
        default=aug_config.get("rotation_max", 8.0),
        help="Maximale Rotation in Grad",
    )
    parser.add_argument(
        "--perspective",
        type=float,
        default=aug_config.get("perspective", 0.05),
        help="Perspektivische Verzerrung (0.0-0.2)",
    )
    parser.add_argument(
        "--shadow",
        type=float,
        default=aug_config.get("shadow", 0.3),
        help="Schatten-Intensität (0.0-1.0)",
    )
    parser.add_argument(
        "--saturation_min",
        type=float,
        default=aug_config.get("saturation_min", 0.6),
        help="Minimaler Sättigungsfaktor",
    )
    parser.add_argument(
        "--saturation_max",
        type=float,
        default=aug_config.get("saturation_max", 1.2),
        help="Maximaler Sättigungsfaktor",
    )
    parser.add_argument(
        "--color_temperature_min",
        type=float,
        default=aug_config.get("color_temperature_min", 0.8),
        help="Minimaler Farbtemperatur-Faktor (kühler)",
    )
    parser.add_argument(
        "--color_temperature_max",
        type=float,
        default=aug_config.get("color_temperature_max", 1.2),
        help="Maximaler Farbtemperatur-Faktor (wärmer)",
    )
    parser.add_argument(
        "--hue_shift_max",
        type=float,
        default=aug_config.get("hue_shift_max", 10.0),
        help="Maximale Farbton-Verschiebung in Grad",
    )
    parser.add_argument(
        "--background_color",
        type=str,
        default=aug_config.get("background_color", "white"),
        help="Hintergrundfarbe: white oder black",
    )

    args = parser.parse_args()
    if args.num_augmentations < 1:
        print("[INFO] num_augmentations < 1 -> setze auf 1")
        args.num_augmentations = 1
    args.jpg_quality = max(1, min(100, args.jpg_quality))

    camera_params = {
        "brightness_range": (args.brightness_min, args.brightness_max),
        "contrast_range": (args.contrast_min, args.contrast_max),
        "saturation_range": (args.saturation_min, args.saturation_max),
        "color_temperature_range": (args.color_temperature_min, args.color_temperature_max),
        "hue_shift_max": args.hue_shift_max,
        "blur_range": (0.0, args.blur_max),
        "noise_range": (0, args.noise_max),
        "rotation_range": (-args.rotation_max, args.rotation_max),
        "perspective": args.perspective,
        "shadow": args.shadow,
        "background_color": args.background_color,
    }

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists():
        print(f"❌ Input-Verzeichnis nicht gefunden: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.png")))
    if not image_files:
        print(f"❌ Keine Bilddateien gefunden in: {input_path}")
        return

    card_sources: Dict[str, Tuple[Optional[str], Path]] = {}
    for img in image_files:
        meta = parse_scryfall_filename(img.name)
        if not meta:
            continue
        card_uuid = meta[0]
        set_code = meta[1]
        card_sources.setdefault(card_uuid, (set_code, img))

    print(f"📋 Gefunden: {len(card_sources)} eindeutige Karten in {input_path}")
    print(f"🎯 Erstelle {args.num_augmentations} Kamera-Augmentierungen pro Karte")
    print(
        f"📋 Camera-Parameter: Helligkeit={camera_params['brightness_range']}, "
        f"Kontrast={camera_params['contrast_range']}, Blur=0-{args.blur_max}, "
        f"Rauschen=0-{args.noise_max}, Rotation=±{args.rotation_max}°"
    )
    print(f"🎨 Hintergrundfarbe: {args.background_color}")

    augmentor = CameraLikeAugmentor(**camera_params)
    crop_cfg = load_symbol_crop_cfg()

    total_variants = 0
    skipped = 0

    for card_uuid, (set_code, img_file) in tqdm(card_sources.items(), desc="Augmentiere Karten", unit="card"):
        if set_code:
            card_dir = output_path / set_code / card_uuid
        else:
            card_dir = output_path / card_uuid
        full_dir = card_dir / "full"
        symbol_dir = card_dir / "symbol"

        if card_dir.exists():
            if not args.overwrite:
                skipped += 1
                continue
            shutil.rmtree(card_dir)

        full_dir.mkdir(parents=True, exist_ok=True)
        if args.save_symbol_crops:
            symbol_dir.mkdir(parents=True, exist_ok=True)

        try:
            original_img = Image.open(img_file).convert("RGB")
            augmentations = augmentor.create_camera_like_augmentations(
                original_img, num_augmentations=args.num_augmentations
            )
            if len(augmentations) <= 1:
                print(f"[WARN] Keine Augmentierungen für {img_file.name}")
                continue

            for aug_idx, aug_img in enumerate(augmentations[1:]):
                filename = f"{card_uuid}_aug_{aug_idx:02d}.jpg"
                full_path = full_dir / filename
                aug_img.save(full_path, "JPEG", quality=args.jpg_quality)

                if args.save_symbol_crops:
                    crop_img = crop_set_symbol(aug_img, crop_cfg)
                    crop_img.save(symbol_dir / filename, "JPEG", quality=args.jpg_quality)
                total_variants += 1
        except Exception as exc:
            print(f"❌ Fehler bei {img_file.name}: {exc}")

    print(
        f"\n✅ Fertig! {total_variants} augmentierte Varianten gespeichert in {output_path} "
        f"(übersprungen: {skipped})"
    )
    if args.save_symbol_crops:
        print("🧩 Symbol-Crops wurden parallel abgelegt.")


if __name__ == "__main__":
    main()
