#!/usr/bin/env python3
"""
Script zur Augmentierung von MTG-Karten f??r das Training.
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
    """L??dt die Konfiguration aus YAML-Datei"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"?s????  Config-Datei nicht gefunden: {config_path}")
        return {}


class CameraLikeAugmentor:
    """Klasse f?r realistische Kamera-Augmentierung"""

    def __init__(self, **params):
        def _resolve_range(range_key, min_key, max_key, default):
            if range_key in params:
                rng = params[range_key]
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    return float(rng[0]), float(rng[1])
            return float(params.get(min_key, default[0])), float(params.get(max_key, default[1]))

        self.brightness_range = _resolve_range('brightness_range', 'brightness_min', 'brightness_max', (0.7, 1.3))
        self.contrast_range = _resolve_range('contrast_range', 'contrast_min', 'contrast_max', (0.8, 1.3))
        self.saturation_range = _resolve_range('saturation_range', 'saturation_min', 'saturation_max', (0.7, 1.2))
        if 'color_temperature_range' in params:
            rng = params['color_temperature_range']
            self.color_temperature_range = (float(rng[0]), float(rng[1]))
        else:
            shift = params.get('color_temp_shift_max', 0.1)
            self.color_temperature_range = (1.0 - shift, 1.0 + shift)
        self.hue_shift_max = float(params.get('hue_shift_max', 10.0))
        self.sharpness_range = _resolve_range('sharpness_range', 'sharpness_min', 'sharpness_max', (0.5, 1.5))
        self.blur_range = _resolve_range('blur_range', 'blur_sigma_min', 'blur_sigma_max', (0.0, 2.0))
        noise_std_max = params.get('noise_std_max')
        if noise_std_max is None:
            legacy_noise = params.get('noise_range', (0, 15))
            noise_std_max = legacy_noise[1] / 255.0
        self.noise_std_max = float(noise_std_max)
        tilt = params.get('tilt_deg_max')
        if tilt is not None:
            self.rotation_range = (-float(tilt), float(tilt))
        else:
            self.rotation_range = params.get('rotation_range', (-8.0, 8.0))
        self.perspective_strength = float(params.get('perspective_max', params.get('perspective', 0.05)))
        self.shadow_strength = float(params.get('shadow', 0.3))
        self.gamma_range = _resolve_range('gamma_range', 'gamma_min', 'gamma_max', (0.9, 1.1))

        background_color = params.get('background_color', 'white')
        self.fill_color = (255, 255, 255) if background_color == 'white' else (0, 0, 0)

        self.brightness_prob = params.get('brightness_prob', 0.8)
        self.contrast_prob = params.get('contrast_prob', 0.7)
        self.gamma_prob = params.get('gamma_prob', 0.5)
        self.saturation_prob = params.get('saturation_prob', 0.6)
        self.color_temperature_prob = params.get('color_temperature_prob', 0.5)
        self.hue_shift_prob = params.get('hue_shift_prob', 0.4)
        self.sharpness_prob = params.get('sharpness_prob', 0.5)
        self.blur_prob = params.get('blur_prob', 0.4)
        self.noise_prob = params.get('noise_prob', 0.3)
        self.rotation_prob = params.get('rotation_prob', 0.6)
        self.perspective_prob = params.get('perspective_prob', 0.2)
        self.shadow_prob = params.get('shadow_prob', 0.3)
    def augment_exposure(self, img: Image.Image) -> Image.Image:
        """Simuliert verschiedene Belichtungen (??ber-/unterbelichtet)"""
        factor = random.uniform(*self.brightness_range)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def augment_contrast(self, img: Image.Image) -> Image.Image:
        """Simuliert unterschiedliche Kontraste"""
        factor = random.uniform(*self.contrast_range)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    def augment_gamma(self, img: Image.Image) -> Image.Image:
        """Simuliert unterschiedliche Gammakurven der Kamera."""
        gamma = random.uniform(*self.gamma_range)
        img_array = np.array(img).astype(np.float32) / 255.0
        corrected = np.power(img_array, 1.0 / max(gamma, 1e-6))
        corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(corrected)
    
    def augment_saturation(self, img: Image.Image) -> Image.Image:
        """Simuliert Farbstiche (Wei?Yabgleich-Probleme)"""
        factor = random.uniform(*self.saturation_range)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)
    
    def augment_sharpness(self, img: Image.Image) -> Image.Image:
        """Simuliert Fokus-Probleme"""
        factor = random.uniform(*self.sharpness_range)
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)
    
    def augment_motion_blur(self, img: Image.Image) -> Image.Image:
        """Simuliert Bewegungsunsch??rfe."""
        sigma = random.uniform(*self.blur_range)
        if sigma > 0.05:
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def augment_noise(self, img: Image.Image) -> Image.Image:
        """Simuliert Sensor-Rauschen bei schlechtem Licht."""
        if self.noise_std_max <= 0:
            return img
        std = random.uniform(0, self.noise_std_max)
        if std <= 0:
            return img
        img_array = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, std, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 1.0)
        return Image.fromarray((noisy_array * 255.0).astype(np.uint8))

    def augment_rotation(self, img: Image.Image) -> Image.Image:
        """Simuliert leichte Kamera-Neigung."""
        angle = random.uniform(*self.rotation_range)
        return img.rotate(angle, expand=False, fillcolor=self.fill_color, resample=Image.BICUBIC)

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
        """Simuliert ungleichm???Yige Beleuchtung/Schatten (Maskenberechnung ist RAM-intensiv)"""
        if random.random() > self.shadow_strength:
            return img
            
        # Erstelle Schatten-Maske
        w, h = img.size
        shadow_mask = Image.new('L', (w, h), 255)
        
        # Zuf??llige Schatten-Position
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
        img_array = np.array(img).astype(np.float32)
        if temp_factor > 1.0:
            img_array[:, :, 0] *= min(temp_factor, 1.4)
            img_array[:, :, 1] *= min(temp_factor * 0.95, 1.2)
            img_array[:, :, 2] *= max(temp_factor * 0.7, 0.8)
        else:
            img_array[:, :, 0] *= max(temp_factor * 0.8, 0.7)
            img_array[:, :, 1] *= max(temp_factor * 0.9, 0.8)
            img_array[:, :, 2] *= min(temp_factor * 1.2, 1.3)
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def augment_hue_shift(self, img: Image.Image) -> Image.Image:
        """Simuliert Farbton-Verschiebungen (Hue-Shift)"""
        if img.mode != 'RGB':
            return img
            
        # Zuf??llige Hue-Verschiebung
        hue_shift = random.uniform(-self.hue_shift_max, self.hue_shift_max)
        
        # Konvertiere RGB zu HSV
        img_hsv = img.convert('HSV')
        hsv_array = np.array(img_hsv).astype(np.float32)
        
        # Verschiebe Hue-Kanal (Index 0)
        hsv_array[:, :, 0] = (hsv_array[:, :, 0] + hue_shift) % 256
        
        # R??ckkonvertierung zu RGB
        img_hsv_shifted = Image.fromarray(hsv_array.astype(np.uint8), mode='HSV')
        return img_hsv_shifted.convert('RGB')
    
    def create_camera_like_augmentations(self, img: Image.Image, num_augmentations: int = 5) -> List[Image.Image]:
        """Erstellt Camera-??hnliche Augmentierungen im Original-Format"""
        augmentations = []
        
        # Original (unver??ndert)
        augmentations.append(img.copy())
        
        # Verschiedene Camera-Bedingungen simulieren
        for i in range(num_augmentations):
            aug_img = img.copy()  # Starte mit Original-Format!
            
            # Realistische Kamera-Effekte anwenden
            if random.random() < self.brightness_prob:
                aug_img = self.augment_exposure(aug_img)
            
            if random.random() < self.contrast_prob:
                aug_img = self.augment_contrast(aug_img)

            if random.random() < self.gamma_prob:
                aug_img = self.augment_gamma(aug_img)
            
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
    
    parser = argparse.ArgumentParser(description='Augmentierung von MTG-Karten f??r Camera-??hnliche Bedingungen')
    parser.add_argument('--input_dir', '-i', type=str, default='./data/scryfall_images',
                        help='Verzeichnis mit Scryfall-Bildern')
    parser.add_argument('--output_dir', '-o', type=str, default='./data/scryfall_augmented',
                        help='Ausgabeverzeichnis f??r augmentierte Bilder')
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
                        help='Maximaler Blur-Radius (Bewegungsunsch??rfe)')
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
                        help='Schatten-Intensit??t (0.0-1.0)')
    parser.add_argument('--saturation_min', type=float, 
                        default=aug_config.get('saturation_min', 0.6),
                        help='Minimaler S??ttigungsfaktor')
    parser.add_argument('--saturation_max', type=float, 
                        default=aug_config.get('saturation_max', 1.2),
                        help='Maximaler S??ttigungsfaktor')
    parser.add_argument('--color_temperature_min', type=float, 
                        default=aug_config.get('color_temperature_min', 0.8),
                        help='Minimaler Farbtemperatur-Faktor (k??hler)')
    parser.add_argument('--color_temperature_max', type=float, 
                        default=aug_config.get('color_temperature_max', 1.2),
                        help='Maximaler Farbtemperatur-Faktor (w??rmer)')
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
        print(f"??O Input-Verzeichnis nicht gefunden: {input_path}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Scryfall-Bilder finden
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    if not image_files:
        print(f"??O Keine Bilddateien gefunden in: {input_path}")
        return

    # Pr??fe, welche Karten bereits augmentiert wurden
    already_augmented = set()
    for f in output_path.glob("*_original.jpg"):
        already_augmented.add(f.stem.replace("_original", ""))

    # Filtere nur neue Karten
    new_image_files = [img for img in image_files if img.stem not in already_augmented]

    print(f"?Y"< Gefunden: {len(image_files)} Scryfall-Bilder, davon {len(new_image_files)} neue Karten")
    print(f"?YZ? Erstelle {args.num_augmentations} Camera-??hnliche Augmentierungen pro Bild (nur neue Karten)")
    print(f"?Y"< Camera-Parameter: Helligkeit={camera_params['brightness_range']}, "
          f"Kontrast={camera_params['contrast_range']}, Blur=0-{args.blur_max}, "
          f"Rauschen=0-{args.noise_max}, Rotation=??{args.rotation_max}??")
    print(f"?YZ? Hintergrundfarbe: {args.background_color}")

    # Augmentor erstellen
    augmentor = CameraLikeAugmentor(**camera_params)

    total_generated = 0

    for img_file in new_image_files:
        print(f"\n?Y"" Verarbeite: {img_file.name}")
        try:
            original_img = Image.open(img_file).convert('RGB')
            print(f"   ?Y"? Original-Format: {original_img.size}")
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
                print(f"   ?Y'? Gespeichert: {output_name} ({aug_img.size})")
                total_generated += 1
        except Exception as e:
            print(f"   ??O Fehler bei {img_file.name}: {e}")

    print(f"\n?o. Fertig! {total_generated} augmentierte Bilder erstellt in {output_path}")
    print(f"?Y", Format: Original Scryfall-Format (488x680) beibehalten")
    print(f"?YZ? Camera-Bedingungen: Belichtung, Kontrast, Blur, Rauschen, Rotation, Perspektive, Schatten")


if __name__ == "__main__":
    main()

