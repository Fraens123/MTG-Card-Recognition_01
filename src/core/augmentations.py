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
        print(f"[WARN] Config-Datei nicht gefunden: {config_path}")
        return {}


class CameraLikeAugmentor:
    """Klasse f?r realistische Kamera-Augmentierung"""

    def __init__(self, **params):
        # Erwartete Keys (werden ueber Config gemappt):
        # brightness/contrast/saturation: [min,max] -> *_range
        # hue_shift_deg: [min,max], color_temperature_range: [min,max], white_balance_shift: [min,max]
        # gamma_range: [min,max], blur_radius: [min,max], blur_prob: float, noise_std_max: float
        # rotation_deg oder tilt_deg_max: float, perspective: float, shadow: float
        # chromatic_aberration_px: float, vignette_strength: float
        # camera_like_strength: [0,1] skaliert Effektstaerke (nicht Wahrscheinlichkeiten)
        def _resolve_range(range_key, min_key, max_key, default):
            if range_key in params:
                rng = params[range_key]
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    return float(rng[0]), float(rng[1])
            return float(params.get(min_key, default[0])), float(params.get(max_key, default[1]))

        # Kamera-nahe Standardwerte (milde Verschiebungen, kein brutaler Blur/Noise).
        self.camera_like_strength = float(np.clip(params.get('camera_like_strength', 1.0), 0.0, 1.0))
        self.brightness_range = _resolve_range('brightness_range', 'brightness_min', 'brightness_max', (0.85, 1.20))
        self.contrast_range = _resolve_range('contrast_range', 'contrast_min', 'contrast_max', (0.85, 1.20))
        self.saturation_range = _resolve_range('saturation_range', 'saturation_min', 'saturation_max', (0.70, 1.20))
        if 'color_temperature_range' in params:
            rng = params['color_temperature_range']
            self.color_temperature_range = (float(rng[0]), float(rng[1]))
        else:
            shift = params.get('color_temp_shift_max', 0.25)
            self.color_temperature_range = (1.0 - shift, 1.0 + shift)
        wb_rng = params.get('white_balance_shift') or params.get('white_balance_shift_range')
        if isinstance(wb_rng, (list, tuple)) and len(wb_rng) == 2:
            self.white_balance_shift_range = (float(wb_rng[0]), float(wb_rng[1]))
        else:
            self.white_balance_shift_range = (-0.10, 0.10)
        self.gamma_range = _resolve_range('gamma_range', 'gamma_min', 'gamma_max', (0.85, 1.15))
        hue_rng = params.get('hue_shift_deg') or params.get('hue_shift_range')
        if isinstance(hue_rng, (list, tuple)) and len(hue_rng) == 2:
            self.hue_shift_range = (float(hue_rng[0]), float(hue_rng[1]))
        else:
            self.hue_shift_range = (-8.0, 8.0)
        self.hue_shift_max = float(params.get('hue_shift_max', max(abs(self.hue_shift_range[0]), abs(self.hue_shift_range[1]))))
        self.sharpness_range = _resolve_range('sharpness_range', 'sharpness_min', 'sharpness_max', (0.85, 1.15))
        # Basis-Range sichern, damit wir sie sp?ter skaliert (camera_like_strength) nutzen k?nnen.
        self._brightness_range_base = self.brightness_range
        self._contrast_range_base = self.contrast_range
        self._saturation_range_base = self.saturation_range
        self._color_temperature_range_base = self.color_temperature_range
        self._gamma_range_base = self.gamma_range
        self._sharpness_range_base = self.sharpness_range
        self._white_balance_shift_base = self.white_balance_shift_range
        self._hue_shift_range_base = self.hue_shift_range

        # Blur/Noise: moderat halten, Effekt skaliert ?ber camera_like_strength.
        blur_rng = params.get('blur_radius') or params.get('blur_radius_range') or params.get('blur_range')
        if isinstance(blur_rng, (list, tuple)) and len(blur_rng) == 2:
            self.blur_radius_min = float(blur_rng[0])
            self.blur_radius_max = float(blur_rng[1])
        else:
            self.blur_radius_min = 0.5
            self.blur_radius_max = 1.5

        noise_std_max = params.get('noise_std_max')
        if noise_std_max is None and 'noise_range' in params:
            legacy_noise = params.get('noise_range', (0, 15))
            if isinstance(legacy_noise, (list, tuple)) and len(legacy_noise) == 2:
                noise_std_max = legacy_noise[1] / 255.0
        if noise_std_max is None:
            noise_std_max = 0.02
        self.noise_std_max = float(noise_std_max)

        # Rotation/Perspektive/Schatten: Effekte ?ber Intensit?t skalieren, nicht ?ber die Wahrscheinlichkeiten.
        tilt_cfg = params.get('tilt_deg_max')
        rotation_cfg = params.get('rotation_deg')
        if rotation_cfg is not None:
            max_rotation_deg = float(rotation_cfg)
        elif tilt_cfg is not None:
            max_rotation_deg = float(tilt_cfg)
        else:
            # Fallback auf evtl. vorhandenes rotation_range, sonst 4?
            rotation_range = params.get('rotation_range', (-4.0, 4.0))
            if isinstance(rotation_range, (list, tuple)) and len(rotation_range) == 2:
                max_rotation_deg = max(abs(rotation_range[0]), abs(rotation_range[1]))
            else:
                max_rotation_deg = 4.0
        self.rotation_base_deg = float(params.get('rotation_base_deg', 0.5))
        self.rotation_max_deg = max(self.rotation_base_deg, max_rotation_deg)

        self.perspective_strength = float(params.get('perspective', params.get('perspective_max', 0.10)))
        self.shadow_strength = float(params.get('shadow', 0.30))
        self.chromatic_aberration_px = float(params.get('chromatic_aberration_px', 1.0))
        self.vignette_strength = float(params.get('vignette_strength', 0.25))

        background_color = params.get('background_color', 'white')
        self.fill_color = (255, 255, 255) if background_color == 'white' else (0, 0, 0)

        # Wahrscheinlichkeiten bleiben unver?ndert vom camera_like_strength.
        self.brightness_prob = params.get('brightness_prob', 0.85)
        self.contrast_prob = params.get('contrast_prob', 0.75)
        self.gamma_prob = params.get('gamma_prob', 0.6)
        self.saturation_prob = params.get('saturation_prob', 0.7)
        self.color_temperature_prob = params.get('color_temperature_prob', 0.7)
        self.hue_shift_prob = params.get('hue_shift_prob', 0.5)
        self.white_balance_prob = params.get('white_balance_prob', 0.7)
        self.sharpness_prob = params.get('sharpness_prob', 0.4)
        self.vignette_prob = params.get('vignette_prob', 0.8)
        self.chromatic_aberration_prob = params.get('chromatic_aberration_prob', 0.8)
        self.blur_prob = params.get('blur_prob', 0.25)
        self.noise_prob = params.get('noise_prob', 0.35)
        self.rotation_prob = params.get('rotation_prob', 0.75)
        self.perspective_prob = params.get('perspective_prob', 0.4)
        self.shadow_prob = params.get('shadow_prob', 0.25)

        self._recompute_effective_strengths()
        # Empfehlung fuer Pi-Cam (warm/gelb, leichte Verzerrung/Vignette):
        # camera_like_strength 1.0, brightness/contrast 0.85-1.20, saturation 0.70-1.20,
        # hue_shift_deg +-8, color_temperature_range 0.75-1.25, white_balance_shift +-0.10,
        # gamma_range 0.85-1.15, rotation_deg ~4, perspective 0.10, shadow 0.30,
        # blur_prob 0.25 mit radius 0.5-1.5, noise_std 0.02, chromatic_aberration_px 1.0, vignette_strength 0.25.

    def _recompute_effective_strengths(self) -> None:
        """Leitet effektive St?rken aus camera_like_strength ab."""
        s = self.camera_like_strength

        def _scale_range(base_range: Tuple[float, float]) -> Tuple[float, float]:
            low, high = base_range
            return (1.0 + (low - 1.0) * s, 1.0 + (high - 1.0) * s)

        self.brightness_range = _scale_range(self._brightness_range_base)
        self.contrast_range = _scale_range(self._contrast_range_base)
        self.saturation_range = _scale_range(self._saturation_range_base)
        self.color_temperature_range = _scale_range(self._color_temperature_range_base)
        self.gamma_range = _scale_range(self._gamma_range_base)
        self.sharpness_range = _scale_range(self._sharpness_range_base)
        self.white_balance_shift_range = tuple(val * s for val in self._white_balance_shift_base)
        self.hue_shift_range = tuple(val * s for val in self._hue_shift_range_base)
        self.hue_shift_max = max(abs(self.hue_shift_range[0]), abs(self.hue_shift_range[1]))

        self.noise_std_max_eff = s * self.noise_std_max
        self.blur_radius_eff_min = self.blur_radius_min * s
        self.blur_radius_eff = max(self.blur_radius_eff_min, s * self.blur_radius_max)
        self.perspective_strength_eff = s * self.perspective_strength
        self.shadow_strength_eff = s * self.shadow_strength
        self.chromatic_aberration_px_eff = s * self.chromatic_aberration_px
        self.vignette_strength_eff = s * self.vignette_strength

        # Rotation linear zwischen Basis und konfiguriertem Maximum.
        deg = self.rotation_base_deg + (self.rotation_max_deg - self.rotation_base_deg) * s
        self.rotation_range = (-deg, deg)

    def _random_apply(self, prob: float) -> bool:
        return random.random() < prob
    def augment_exposure(self, img: Image.Image) -> Image.Image:
        """Simuliert verschiedene Belichtungen (leicht ?ber-/unterbelichtet)."""
        factor = random.uniform(*self.brightness_range)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    def augment_contrast(self, img: Image.Image) -> Image.Image:
        """Simuliert unterschiedliche Kontraste."""
        factor = random.uniform(*self.contrast_range)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    def augment_gamma(self, img: Image.Image) -> Image.Image:
        """Simuliert unterschiedliche Gammakurven der Kamera (kleiner Bereich)."""
        gamma = random.uniform(*self.gamma_range)
        img_array = np.array(img).astype(np.float32) / 255.0
        corrected = np.power(img_array, 1.0 / max(gamma, 1e-6))
        corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(corrected)

    def augment_white_balance(self, img: Image.Image) -> Image.Image:
        """Kleiner globaler White-Balance-Shift (warm/kuehl)."""
        shift = random.uniform(*self.white_balance_shift_range)
        r_gain = 1.0 + shift
        b_gain = 1.0 - shift
        g_gain = 1.0 + 0.5 * shift
        img_array = np.array(img).astype(np.float32)
        img_array[:, :, 0] *= r_gain
        img_array[:, :, 1] *= g_gain
        img_array[:, :, 2] *= b_gain
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def augment_saturation(self, img: Image.Image) -> Image.Image:
        """Simuliert subtile Farbstiche (leichte S?ttigungsschieber)."""
        factor = random.uniform(*self.saturation_range)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)

    def augment_sharpness(self, img: Image.Image) -> Image.Image:
        """Simuliert Fokus-Probleme."""
        factor = random.uniform(*self.sharpness_range)
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)

    def augment_motion_blur(self, img: Image.Image) -> Image.Image:
        """Simuliert milde Bewegungsunsch?rfe (Radius wird ?ber camera_like_strength skaliert)."""
        if self.blur_radius_eff <= 0.01:
            return img
        radius = random.uniform(self.blur_radius_eff_min, self.blur_radius_eff)
        if radius <= 0.05:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def augment_noise(self, img: Image.Image) -> Image.Image:
        """Simuliert Sensor-Rauschen bei schlechtem Licht."""
        if self.noise_std_max_eff <= 0:
            return img
        std = random.uniform(0, self.noise_std_max_eff)
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
        """Simuliert leichte perspektivische Verzerrung (Warp-Transform kostet merklich CPU-Zeit)."""
        if self.perspective_strength_eff <= 0:
            return img

        w, h = img.size
        strength = random.uniform(-self.perspective_strength_eff, self.perspective_strength_eff)
        offset = int(w * strength)

        if abs(offset) > 1:
            new_img = img.transform(
                (w, h),
                Image.PERSPECTIVE,
                (0, offset, w, 0, w, h, 0, h - offset),
                resample=Image.BICUBIC,
                fillcolor=self.fill_color,
            )
            return new_img
        return img

    def augment_shadow(self, img: Image.Image) -> Image.Image:
        """Simuliert ungleichm??Yige Beleuchtung/Schatten (Maskenberechnung ist RAM-intensiv)."""
        if self.shadow_strength_eff <= 0:
            return img

        w, h = img.size
        shadow_mask = Image.new('L', (w, h), 255)

        shadow_x = random.randint(0, max(1, w // 3))
        shadow_y = random.randint(0, max(1, h // 3))
        shadow_w = random.randint(max(8, w // 6), max(12, w // 2))
        shadow_h = random.randint(max(8, h // 6), max(12, h // 2))

        base_intensity = int(180 + 40 * (1.0 - self.shadow_strength_eff))
        shadow_region = Image.new('L', (shadow_w, shadow_h), base_intensity)
        shadow_region = shadow_region.filter(ImageFilter.GaussianBlur(radius=18))
        shadow_mask.paste(shadow_region, (shadow_x, shadow_y))

        img_array = np.array(img).astype(np.float32)
        mask_array = np.array(shadow_mask).astype(np.float32) / 255.0
        shadow_factor = max(0.5, 1.0 - 0.6 * self.shadow_strength_eff)
        mask_array = shadow_factor + (mask_array - shadow_factor) * self.shadow_strength_eff

        img_array *= mask_array[:, :, None]
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def augment_color_temperature(self, img: Image.Image) -> Image.Image:
        """Simuliert realistische Farbtemperatur-Schwankungen (k?hler/w?rmer)."""
        temp_factor = random.uniform(*self.color_temperature_range)
        img_array = np.array(img).astype(np.float32)

        if temp_factor >= 1.0:
            # W?rmer: mehr Rot/Gelb, leicht weniger Blau.
            shift = temp_factor - 1.0
            rgb_scale = np.array([1.0 + 0.8 * shift, 1.0 + 0.35 * shift, 1.0 - 0.7 * shift], dtype=np.float32)
        else:
            # K?hler: mehr Blau, etwas weniger Rot/Gelb.
            shift = 1.0 - temp_factor
            rgb_scale = np.array([1.0 - 0.7 * shift, 1.0 - 0.35 * shift, 1.0 + 0.8 * shift], dtype=np.float32)

        img_array *= rgb_scale[None, None, :]
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def augment_vignette(self, img: Image.Image) -> Image.Image:
        """Vignette/Randabdunklung."""
        if self.vignette_strength_eff <= 0:
            return img
        img_array = np.array(img).astype(np.float32) / 255.0
        h, w, _ = img_array.shape
        y, x = np.ogrid[:h, :w]
        y_center = (h - 1) / 2.0
        x_center = (w - 1) / 2.0
        dist = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
        max_dist = np.sqrt(x_center**2 + y_center**2)
        mask = 1.0 - self.vignette_strength_eff * (dist / (max_dist + 1e-6)) ** 2
        mask = np.clip(mask, 0.0, 1.0)
        img_array *= mask[:, :, None]
        img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def augment_chromatic_aberration(self, img: Image.Image) -> Image.Image:
        """Kanal-Versatz fuer leichte chromatische Aberration."""
        if self.chromatic_aberration_px_eff <= 0:
            return img
        shift = int(round(self.chromatic_aberration_px_eff))
        if shift == 0:
            return img
        arr = np.array(img)

        def _shift_channel(channel: np.ndarray, dx: int, dy: int) -> np.ndarray:
            h, w = channel.shape
            padded = np.pad(channel, ((abs(dy), abs(dy)), (abs(dx), abs(dx))), mode="edge")
            y0 = abs(dy) - dy
            x0 = abs(dx) - dx
            return padded[y0:y0 + h, x0:x0 + w]

        r = _shift_channel(arr[:, :, 0], shift, shift)
        g = arr[:, :, 1]
        b = _shift_channel(arr[:, :, 2], -shift, -shift)
        out = np.stack([r, g, b], axis=2).astype(np.uint8)
        return Image.fromarray(out)

    def augment_hue_shift(self, img: Image.Image) -> Image.Image:
        """Simuliert Farbton-Verschiebungen (Hue-Shift)."""
        if img.mode != 'RGB':
            return img

        max_shift = self.hue_shift_max
        if max_shift <= 0:
            return img
        hue_shift = random.uniform(-max_shift, max_shift)

        img_hsv = img.convert('HSV')
        hsv_array = np.array(img_hsv).astype(np.float32)
        hsv_array[:, :, 0] = (hsv_array[:, :, 0] + hue_shift) % 256
        img_hsv_shifted = Image.fromarray(hsv_array.astype(np.uint8), mode='HSV')
        return img_hsv_shifted.convert('RGB')
    
    def apply(self, img: Image.Image) -> Image.Image:
        """
        Wendet die Camera-Chain in einer kontrollierten Reihenfolge an:
        Belichtung/Kontrast/Gamma -> WB/Temperatur/Hue/Saettigung -> Vignette/CA -> Rotation/Perspektive -> Blur/Noise.
        """
        out = img.copy()

        if self._random_apply(self.brightness_prob):
            out = self.augment_exposure(out)
        if self._random_apply(self.contrast_prob):
            out = self.augment_contrast(out)
        if self._random_apply(self.gamma_prob):
            out = self.augment_gamma(out)

        if self._random_apply(self.white_balance_prob):
            out = self.augment_white_balance(out)
        if self._random_apply(self.color_temperature_prob):
            out = self.augment_color_temperature(out)
        if self._random_apply(self.hue_shift_prob):
            out = self.augment_hue_shift(out)
        if self._random_apply(self.saturation_prob):
            out = self.augment_saturation(out)
        if self._random_apply(self.sharpness_prob):
            out = self.augment_sharpness(out)

        if self._random_apply(self.vignette_prob):
            out = self.augment_vignette(out)
        if self._random_apply(self.shadow_prob):
            out = self.augment_shadow(out)
        if self._random_apply(self.chromatic_aberration_prob):
            out = self.augment_chromatic_aberration(out)

        if self._random_apply(self.rotation_prob):
            out = self.augment_rotation(out)
        if self._random_apply(self.perspective_prob):
            out = self.augment_perspective(out)

        if self._random_apply(self.blur_prob):
            out = self.augment_motion_blur(out)
        if self._random_apply(self.noise_prob):
            out = self.augment_noise(out)
        return out

    __call__ = apply

    def create_camera_like_augmentations(self, img: Image.Image, num_augmentations: int = 5) -> List[Image.Image]:
        """Erstellt Camera-??hnliche Augmentierungen im Original-Format."""
        augmentations = [img.copy()]
        for _ in range(num_augmentations):
            augmentations.append(self.apply(img))
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
                        default=aug_config.get('brightness_min', 0.9),
                        help='Minimaler Helligkeitsfaktor (Belichtung)')
    parser.add_argument('--brightness_max', type=float, 
                        default=aug_config.get('brightness_max', 1.1),
                        help='Maximaler Helligkeitsfaktor (Belichtung)')
    parser.add_argument('--contrast_min', type=float, 
                        default=aug_config.get('contrast_min', 0.9),
                        help='Minimaler Kontrastfaktor')
    parser.add_argument('--contrast_max', type=float, 
                        default=aug_config.get('contrast_max', 1.1),
                        help='Maximaler Kontrastfaktor')
    parser.add_argument('--blur_max', type=float, 
                        default=aug_config.get('blur_max', 1.5),
                        help='Maximaler Blur-Radius (Bewegungsunsch??rfe)')
    parser.add_argument('--noise_max', type=float, 
                        default=aug_config.get('noise_max', 5.0),
                        help='Maximales Sensor-Rauschen (0-255 Std-Abweichung)')
    parser.add_argument('--rotation_max', type=float, 
                        default=aug_config.get('rotation_max', 3.0),
                        help='Maximale Rotation in Grad')
    parser.add_argument('--perspective', type=float, 
                        default=aug_config.get('perspective', 0.06),
                        help='Perspektivische Verzerrung (0.0-0.2)')
    parser.add_argument('--shadow', type=float, 
                        default=aug_config.get('shadow', 0.25),
                        help='Schatten-Intensit??t (0.0-1.0)')
    parser.add_argument('--saturation_min', type=float, 
                        default=aug_config.get('saturation_min', 0.9),
                        help='Minimaler S??ttigungsfaktor')
    parser.add_argument('--saturation_max', type=float, 
                        default=aug_config.get('saturation_max', 1.1),
                        help='Maximaler S??ttigungsfaktor')
    parser.add_argument('--color_temperature_min', type=float, 
                        default=aug_config.get('color_temperature_min', 0.93),
                        help='Minimaler Farbtemperatur-Faktor (k??hler)')
    parser.add_argument('--color_temperature_max', type=float, 
                        default=aug_config.get('color_temperature_max', 1.07),
                        help='Maximaler Farbtemperatur-Faktor (w??rmer)')
    parser.add_argument('--hue_shift_max', type=float, 
                        default=aug_config.get('hue_shift_max', 8.0),
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
        print(f"[ERROR] Input-Verzeichnis nicht gefunden: {input_path}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Scryfall-Bilder finden
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    if not image_files:
        print(f"[ERROR] Keine Bilddateien gefunden in: {input_path}")
        return

    # Pr??fe, welche Karten bereits augmentiert wurden
    already_augmented = set()
    for f in output_path.glob("*_original.jpg"):
        already_augmented.add(f.stem.replace("_original", ""))

    # Filtere nur neue Karten
    new_image_files = [img for img in image_files if img.stem not in already_augmented]

    print(
        f"[INFO] Gefunden: {len(image_files)} Scryfall-Bilder, davon "
        f"{len(new_image_files)} neue Karten"
    )
    print(
        f"[INFO] Erstelle {args.num_augmentations} Camera-aehnliche Augmentierungen pro Bild "
        "(nur neue Karten)"
    )
    print(
        "[INFO] Camera-Parameter: "
        f"Helligkeit={camera_params['brightness_range']}, "
        f"Kontrast={camera_params['contrast_range']}, "
        f"Blur=0-{args.blur_max}, "
        f"Rauschen=0-{args.noise_max}, "
        f"Rotation=+/-{args.rotation_max}"
    )
    print(f"[INFO] Hintergrundfarbe: {args.background_color}")

    # Augmentor erstellen
    augmentor = CameraLikeAugmentor(**camera_params)

    total_generated = 0

    for img_file in new_image_files:
        print(f"\n[RUN] Verarbeite: {img_file.name}")
        try:
            original_img = Image.open(img_file).convert('RGB')
            print(f"   [INFO] Original-Format: {original_img.size}")
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
                print(f"   [SAVE] {output_name} ({aug_img.size})")
                total_generated += 1
        except Exception as e:
            print(f"   [ERROR] Fehler bei {img_file.name}: {e}")

    print(f"\n[OK] Fertig! {total_generated} augmentierte Bilder erstellt in {output_path}")
    print("[INFO] Format: Original Scryfall-Format (488x680) beibehalten")
    print("[INFO] Camera-Bedingungen: Belichtung, Kontrast, Blur, Rauschen, Rotation, Perspektive, Schatten")


if __name__ == "__main__":
    main()

