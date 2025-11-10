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
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from typing import List, Tuple


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
        """Simuliert Bewegungsunschärfe (teuer: großer GaussianBlur läuft komplett auf der CPU)"""
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
        """Simuliert leichte perspektivische Verzerrung (Warp-Transform kostet merklich CPU-Zeit)"""
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
        """Simuliert ungleichmäßige Beleuchtung/Schatten (Maskenberechnung ist RAM-intensiv)"""
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
    
    parser = argparse.ArgumentParser(description='Augmentierung von MTG-Karten für Camera-ähnliche Bedingungen')
    parser.add_argument('--input_dir', '-i', type=str, default='./data/scryfall_images',
                        help='Verzeichnis mit Scryfall-Bildern')
    parser.add_argument('--output_dir', '-o', type=str, default='./data/scryfall_augmented',
                        help='Ausgabeverzeichnis für augmentierte Bilder')
    parser.add_argument('--num_augmentations', '-n', type=int, 
                        default=aug_config.get('num_augmentations', 8),
                        help='Anzahl Augmentierungen pro Bild')
    
    # Camera-Parameter mit Defaults aus Konfiguration
    parser.add_argument('--brightness_min', type=float, 
                        default=aug_config.get('brightness_min', 0.6),
                        help='Minimaler Helligkeitsfaktor (Belichtung)')
    parser.add_argument('--brightness_max', type=float, 
                        default=aug_config.get('brightness_max', 1.4),
                        help='Maximaler Helligkeitsfaktor (Belichtung)')
    parser.add_argument('--contrast_min', type=float, 
                        default=aug_config.get('contrast_min', 0.7),
                        help='Minimaler Kontrastfaktor')
    parser.add_argument('--contrast_max', type=float, 
                        default=aug_config.get('contrast_max', 1.3),
                        help='Maximaler Kontrastfaktor')
    parser.add_argument('--blur_max', type=float, 
                        default=aug_config.get('blur_max', 2.0),
                        help='Maximaler Blur-Radius (Bewegungsunschärfe)')
    parser.add_argument('--noise_max', type=float, 
                        default=aug_config.get('noise_max', 20.0),
                        help='Maximales Sensor-Rauschen')
    parser.add_argument('--rotation_max', type=float, 
                        default=aug_config.get('rotation_max', 8.0),
                        help='Maximale Rotation in Grad')
    parser.add_argument('--perspective', type=float, 
                        default=aug_config.get('perspective', 0.05),
                        help='Perspektivische Verzerrung (0.0-0.2)')
    parser.add_argument('--shadow', type=float, 
                        default=aug_config.get('shadow', 0.3),
                        help='Schatten-Intensität (0.0-1.0)')
    parser.add_argument('--saturation_min', type=float, 
                        default=aug_config.get('saturation_min', 0.6),
                        help='Minimaler Sättigungsfaktor')
    parser.add_argument('--saturation_max', type=float, 
                        default=aug_config.get('saturation_max', 1.2),
                        help='Maximaler Sättigungsfaktor')
    parser.add_argument('--color_temperature_min', type=float, 
                        default=aug_config.get('color_temperature_min', 0.8),
                        help='Minimaler Farbtemperatur-Faktor (kühler)')
    parser.add_argument('--color_temperature_max', type=float, 
                        default=aug_config.get('color_temperature_max', 1.2),
                        help='Maximaler Farbtemperatur-Faktor (wärmer)')
    parser.add_argument('--hue_shift_max', type=float, 
                        default=aug_config.get('hue_shift_max', 10.0),
                        help='Maximale Farbton-Verschiebung in Grad')
    parser.add_argument('--background_color', type=str, 
                        default=aug_config.get('background_color', 'white'),
                        help='Hintergrundfarbe: white oder black')
    
    args = parser.parse_args()
    
    # Camera-Parameter zusammenstellen
    camera_params = {
        'brightness_range': (args.brightness_min, args.brightness_max),
        'contrast_range': (args.contrast_min, args.contrast_max),
        'saturation_range': (args.saturation_min, args.saturation_max),
        'color_temperature_range': (args.color_temperature_min, args.color_temperature_max),
        'hue_shift_max': args.hue_shift_max,
        'blur_range': (0.0, args.blur_max),
        'noise_range': (0, args.noise_max),
        'rotation_range': (-args.rotation_max, args.rotation_max),
        'perspective': args.perspective,
        'shadow': args.shadow,
        'background_color': args.background_color,
    }
    
    # Verzeichnisse einrichten
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if not input_path.exists():
        print(f"❌ Input-Verzeichnis nicht gefunden: {input_path}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Scryfall-Bilder finden
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    if not image_files:
        print(f"❌ Keine Bilddateien gefunden in: {input_path}")
        return

    # Prüfe, welche Karten bereits augmentiert wurden
    already_augmented = set()
    for f in output_path.glob("*_original.jpg"):
        already_augmented.add(f.stem.replace("_original", ""))

    # Filtere nur neue Karten
    new_image_files = [img for img in image_files if img.stem not in already_augmented]

    print(f"📋 Gefunden: {len(image_files)} Scryfall-Bilder, davon {len(new_image_files)} neue Karten")
    print(f"🎯 Erstelle {args.num_augmentations} Camera-ähnliche Augmentierungen pro Bild (nur neue Karten)")
    print(f"📋 Camera-Parameter: Helligkeit={camera_params['brightness_range']}, "
          f"Kontrast={camera_params['contrast_range']}, Blur=0-{args.blur_max}, "
          f"Rauschen=0-{args.noise_max}, Rotation=±{args.rotation_max}°")
    print(f"🎨 Hintergrundfarbe: {args.background_color}")

    # Augmentor erstellen
    augmentor = CameraLikeAugmentor(**camera_params)

    total_generated = 0

    for img_file in new_image_files:
        print(f"\n🔄 Verarbeite: {img_file.name}")
        try:
            original_img = Image.open(img_file).convert('RGB')
            print(f"   📐 Original-Format: {original_img.size}")
            augmentations = augmentor.create_camera_like_augmentations(
                original_img, 
                num_augmentations=args.num_augmentations
            )
            card_name = img_file.stem
            for i, aug_img in enumerate(augmentations):
                if i == 0:
                    output_name = f"{card_name}_original.jpg"
                else:
                    output_name = f"{card_name}_aug_{i:02d}.jpg"
                output_file = output_path / output_name
                aug_img.save(output_file, 'JPEG', quality=85)
                print(f"   💾 Gespeichert: {output_name} ({aug_img.size})")
                total_generated += 1
        except Exception as e:
            print(f"   ❌ Fehler bei {img_file.name}: {e}")

    print(f"\n✅ Fertig! {total_generated} augmentierte Bilder erstellt in {output_path}")
    print(f"📂 Format: Original Scryfall-Format (488x680) beibehalten")
    print(f"🎭 Camera-Bedingungen: Belichtung, Kontrast, Blur, Rauschen, Rotation, Perspektive, Schatten")


if __name__ == "__main__":
    main()
