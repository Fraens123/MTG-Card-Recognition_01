#!/usr/bin/env python3
"""
Camera Image Analyzer - Analysiert echte Pi Cam Bilder für optimale Augmentierung

Dieses Script analysiert die Camera-Bilder im data/camera_images Ordner und 
berechnet optimale Augmentierungsparameter basierend auf echten Kamera-Eigenschaften.

Usage:
    python src/cardscanner/analyze_camera_images.py
"""

import os
import cv2
import numpy as np
import yaml
from PIL import Image, ImageStat, ImageFilter, ImageEnhance
import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import json
import colorsys

# Import fallbacks für Play-Button Kompatibilität
try:
    from .config import load_config
except ImportError:
    try:
        from config import load_config
    except ImportError:
        def load_config():
            with open('config.yaml', 'r') as f:
                return yaml.safe_load(f)

class CameraImageAnalyzer:
    """Analysiert Camera-Bilder für optimale Augmentierungs-Parameter"""
    
    def __init__(self):
        self.brightness_values = []
        self.contrast_values = []
        self.blur_values = []
        self.noise_values = []
        self.rotation_values = []
        self.shadow_values = []
        self.saturation_values = []        # NEU: Farbsättigung
        self.color_temperature_values = [] # NEU: Farbtemperatur
        self.hue_values = []              # NEU: Farbton-Verschiebungen
        
    def analyze_brightness_contrast(self, image_path):
        """Analysiert Helligkeit und Kontrast"""
        try:
            with Image.open(image_path) as img:
                # Zu RGB konvertieren
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Brightness (Durchschnitt aller Pixel)
                stat = ImageStat.Stat(img)
                brightness = sum(stat.mean) / len(stat.mean) / 255.0  # Normalisiert 0-1
                
                # Contrast (Standardabweichung)
                contrast = sum(stat.stddev) / len(stat.stddev) / 255.0  # Normalisiert 0-1
                
                return brightness, contrast
        except Exception as e:
            print(f"❌ Fehler bei Brightness/Contrast Analyse von {image_path}: {e}")
            return 0.7, 0.2  # Default Werte
    
    def analyze_blur(self, image_path):
        """Analysiert Bewegungsunschärfe mit Laplacian Variance"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return 0.0
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Laplacian Variance - niedrige Werte = mehr Blur
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalisiere zu Blur-Faktor (0-10, höher = mehr blur)
            # Typische Werte: scharf > 500, leicht blur 100-500, stark blur < 100
            if laplacian_var > 500:
                blur_factor = 0.0
            elif laplacian_var > 100:
                blur_factor = (500 - laplacian_var) / 400 * 2.0
            else:
                blur_factor = 2.0 + (100 - laplacian_var) / 100 * 3.0
                
            return min(blur_factor, 5.0)  # Max 5.0 blur
        except Exception as e:
            print(f"❌ Fehler bei Blur Analyse von {image_path}: {e}")
            return 1.0  # Default Wert
    
    def analyze_noise(self, image_path):
        """Analysiert Sensor-Rauschen"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return 0.0
                
            # Konvertiere zu Graustufen
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Schätze Rauschen durch lokale Standardabweichung
            # Verwende 3x3 Filter für lokale Varianz
            kernel = np.ones((3, 3), np.float32) / 9
            mean_filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            noise_estimate = np.std(gray.astype(np.float32) - mean_filtered)
            
            # Normalisiere zu 0-50 Range (typisch für noise_max Parameter)
            return min(noise_estimate, 50.0)
        except Exception as e:
            print(f"❌ Fehler bei Noise Analyse von {image_path}: {e}")
            return 10.0  # Default Wert
    
    def analyze_rotation(self, image_path):
        """Analysiert Bildrotation durch Kanten-Detection"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return 0.0
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Canny Edge Detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Hough Line Transform für dominante Linien
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for line in lines[:10]:  # Nur erste 10 Linien analysieren
                    rho, theta = line[0]
                    angle = (theta * 180 / np.pi) - 90  # Konvertiere zu Grad, 0° = vertikal
                    # Normalisiere zu -90 bis +90 Grad
                    while angle > 90:
                        angle -= 180
                    while angle < -90:
                        angle += 180
                    angles.append(abs(angle))
                
                if angles:
                    # Median Rotation (robust gegen Outliers)
                    rotation = np.median(angles)
                    return min(rotation, 15.0)  # Max 15° Rotation
            
            return 1.0  # Default wenn keine Linien erkannt
        except Exception as e:
            print(f"❌ Fehler bei Rotation Analyse von {image_path}: {e}")
            return 1.0  # Default Wert
    
    def analyze_shadow_variations(self, image_path):
        """Analysiert Schatten und Licht-Variationen"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Konvertiere zu numpy array
                img_array = np.array(img)
                
                # Berechne lokale Helligkeit in 4x4 Grid
                h, w = img_array.shape[:2]
                grid_h, grid_w = h // 4, w // 4
                
                brightness_grid = []
                for i in range(4):
                    for j in range(4):
                        y1, y2 = i * grid_h, (i + 1) * grid_h
                        x1, x2 = j * grid_w, (j + 1) * grid_w
                        region = img_array[y1:y2, x1:x2]
                        region_brightness = np.mean(region) / 255.0
                        brightness_grid.append(region_brightness)
                
                # Schatten-Faktor = Standardabweichung der Helligkeit
                shadow_factor = np.std(brightness_grid)
                return min(shadow_factor, 0.5)  # Max 0.5 shadow
                
        except Exception as e:
            print(f"❌ Fehler bei Shadow Analyse von {image_path}: {e}")
            return 0.2  # Default Wert

    def analyze_saturation(self, image_path):
        """Analysiert Farbsättigung der Camera-Bilder (OPTIMIERT)"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Konvertiere zu numpy array und sample nur jeden 10. Pixel für Performance
                img_array = np.array(img)
                h, w = img_array.shape[:2]
                sampled = img_array[::10, ::10]  # Nur jeden 10. Pixel
                
                # Berechne Sättigung über RGB-Werte (schneller als HSV)
                r = sampled[:, :, 0].astype(float)
                g = sampled[:, :, 1].astype(float) 
                b = sampled[:, :, 2].astype(float)
                
                # Max und Min für Sättigungs-Berechnung
                rgb_max = np.maximum(np.maximum(r, g), b)
                rgb_min = np.minimum(np.minimum(r, g), b)
                
                # Sättigung = (Max - Min) / Max (außer wenn Max = 0)
                saturation = np.divide(rgb_max - rgb_min, rgb_max, out=np.zeros_like(rgb_max), where=rgb_max!=0)
                
                # Filtere sehr dunkle Pixel (diese haben unzuverlässige Sättigung)
                brightness = rgb_max / 255.0
                valid_pixels = brightness > 0.1
                
                if np.any(valid_pixels):
                    mean_saturation = np.mean(saturation[valid_pixels])
                else:
                    mean_saturation = np.mean(saturation)
                
                return mean_saturation
                
        except Exception as e:
            print(f"❌ Fehler bei Sättigungs-Analyse von {image_path}: {e}")
            return 0.5  # Default Wert

    def analyze_color_temperature(self, image_path):
        """Analysiert Farbtemperatur (Blau/Orange Balance) der Camera-Bilder"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                
                # Berechne R/G/B Durchschnittswerte
                r_mean = np.mean(img_array[:, :, 0])
                g_mean = np.mean(img_array[:, :, 1])
                b_mean = np.mean(img_array[:, :, 2])
                
                # Farbtemperatur-Index: (R + G) / (2 * B)
                # > 1.0 = warm (gelblich/rötlich)
                # < 1.0 = kalt (bläulich)
                if b_mean > 10:  # Vermeide Division durch 0
                    color_temp_index = (r_mean + g_mean) / (2 * b_mean)
                else:
                    color_temp_index = 1.0
                
                return color_temp_index
                
        except Exception as e:
            print(f"❌ Fehler bei Farbtemperatur-Analyse von {image_path}: {e}")
            return 1.0  # Neutral

    def analyze_hue_shift(self, image_path):
        """Analysiert Farbton-Verschiebungen der Camera-Bilder (OPTIMIERT)"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Sample nur jeden 20. Pixel für Performance
                img_array = np.array(img)
                sampled = img_array[::20, ::20]
                
                r = sampled[:, :, 0].astype(float) / 255.0
                g = sampled[:, :, 1].astype(float) / 255.0
                b = sampled[:, :, 2].astype(float) / 255.0
                
                # Schnelle HSV-Berechnung für dominante Farbe
                rgb_max = np.maximum(np.maximum(r, g), b)
                rgb_min = np.minimum(np.minimum(r, g), b)
                diff = rgb_max - rgb_min
                
                # Nur farbige Pixel verwenden (hohe Sättigung)
                saturation = np.divide(diff, rgb_max, out=np.zeros_like(rgb_max), where=rgb_max!=0)
                colored_pixels = (saturation > 0.3) & (rgb_max > 0.2) & (rgb_max < 0.8)
                
                if np.any(colored_pixels):
                    # Vereinfachte Farbton-Berechnung (nur grober Bereich)
                    r_dom = np.median(r[colored_pixels])
                    g_dom = np.median(g[colored_pixels])
                    b_dom = np.median(b[colored_pixels])
                    
                    # Grobe Farbton-Klassifikation
                    if r_dom > g_dom and r_dom > b_dom:
                        if g_dom > b_dom:
                            hue_shift = 30  # Orange/Gelb
                        else:
                            hue_shift = -30 # Magenta/Rot
                    elif g_dom > r_dom and g_dom > b_dom:
                        hue_shift = 0   # Grün (Referenz)
                    else:
                        hue_shift = -60  # Blau
                        
                    return hue_shift
                else:
                    return 0.0  # Keine farbigen Pixel
                
        except Exception as e:
            print(f"❌ Fehler bei Farbton-Analyse von {image_path}: {e}")
            return 0.0

    def analyze_brightness_range(self, image_path):
        """Erweiterte Helligkeits-Analyse mit besserer Multiplikator-Berechnung"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # PIL ImageStat für bessere Statistiken
                stat = ImageStat.Stat(img)
                
                # Durchschnittliche Helligkeit (0-255)
                avg_brightness = sum(stat.mean) / 3.0
                
                # Normalisiere auf 0-1
                brightness_norm = avg_brightness / 255.0
                
                # Berechne optimale Multiplikatoren für Augmentierung
                # Wenn Bild zu hell (>0.7): mehr Dunkel-Augmentierung
                # Wenn Bild zu dunkel (<0.4): mehr Hell-Augmentierung
                if brightness_norm > 0.7:
                    # Helle Bilder: 0.6 - 1.1 (mehr abdunkeln)
                    brightness_factor = brightness_norm * 1.2  # Korrektur
                    min_mult = max(0.6, brightness_factor - 0.3)
                    max_mult = min(1.1, brightness_factor + 0.1)
                elif brightness_norm < 0.4:
                    # Dunkle Bilder: 0.9 - 1.4 (mehr aufhellen)
                    brightness_factor = brightness_norm * 1.5
                    min_mult = max(0.9, brightness_factor)
                    max_mult = min(1.4, brightness_factor + 0.5)
                else:
                    # Normale Bilder: 0.8 - 1.2 (ausgewogen)
                    min_mult = 0.8
                    max_mult = 1.2
                
                return {
                    'brightness_norm': brightness_norm,
                    'min_multiplier': min_mult,
                    'max_multiplier': max_mult
                }
                
        except Exception as e:
            print(f"❌ Fehler bei erweiterter Helligkeits-Analyse von {image_path}: {e}")
            return {'brightness_norm': 0.5, 'min_multiplier': 0.8, 'max_multiplier': 1.2}
    
    def analyze_directory(self, camera_dir):
        """Analysiert alle Bilder im Camera-Verzeichnis"""
        camera_path = Path(camera_dir)
        
        if not camera_path.exists():
            print(f"❌ Camera-Verzeichnis nicht gefunden: {camera_dir}")
            return
        
        # Unterstützte Bildformate
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in camera_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"❌ Keine Bilder im Verzeichnis gefunden: {camera_dir}")
            return
        
        print(f"📷 Analysiere {len(image_files)} Camera-Bilder...")
        print(f"📂 Verzeichnis: {camera_dir}")
        print()
        
        # Reset der Listen
        self.brightness_values = []
        self.contrast_values = []
        self.blur_values = []
        self.noise_values = []
        self.rotation_values = []
        self.shadow_values = []
        self.saturation_values = []
        self.color_temperature_values = []
        self.hue_values = []
        
        for i, image_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] 🔍 Analysiere: {image_file.name}")
            
            try:
                # Basis-Analysen
                brightness, contrast = self.analyze_brightness_contrast(image_file)
                blur = self.analyze_blur(image_file)
                noise = self.analyze_noise(image_file)
                rotation = self.analyze_rotation(image_file)
                shadow = self.analyze_shadow_variations(image_file)
                
                # Neue erweiterte Analysen
                saturation = self.analyze_saturation(image_file)
                color_temp = self.analyze_color_temperature(image_file)
                hue_shift = self.analyze_hue_shift(image_file)
                brightness_range = self.analyze_brightness_range(image_file)
                
                # Werte sammeln
                self.brightness_values.append(brightness)
                self.contrast_values.append(contrast)
                self.blur_values.append(blur)
                self.noise_values.append(noise)
                self.rotation_values.append(rotation)
                self.shadow_values.append(shadow)
                self.saturation_values.append(saturation)
                self.color_temperature_values.append(color_temp)
                self.hue_values.append(hue_shift)
                
                # Ausgabe - erweitert
                print(f"   💡 Helligkeit: {brightness:.3f}, 🎭 Kontrast: {contrast:.3f}")
                print(f"   🌀 Blur: {blur:.1f}, 📊 Rauschen: {noise:.1f}")
                print(f"   🔄 Rotation: {rotation:.1f}°, 🌙 Schatten: {shadow:.3f}")
                print(f"   🎨 Sättigung: {saturation:.3f}, 🌡️ Farbtemp: {color_temp:.3f}")
                print(f"   🌈 Farbton: {hue_shift:+.1f}°, 📈 Hellbereich: {brightness_range['min_multiplier']:.2f}-{brightness_range['max_multiplier']:.2f}")
                print()
                
            except Exception as e:
                print(f"❌ Fehler beim Analysieren von {image_file}: {e}")
                print()
        
        print("✅ Analyse abgeschlossen!")
    
    def calculate_optimal_parameters(self):
        """Berechnet optimale Augmentierungsparameter basierend auf der Analyse (ERWEITERT)"""
        if not self.brightness_values:
            print("❌ Keine Analysedaten verfügbar!")
            return None
        
        # Basis-Statistiken berechnen
        brightness_stats = {
            'mean': np.mean(self.brightness_values),
            'std': np.std(self.brightness_values),
            'min': np.min(self.brightness_values),
            'max': np.max(self.brightness_values)
        }
        
        contrast_stats = {
            'mean': np.mean(self.contrast_values),
            'std': np.std(self.contrast_values),
            'min': np.min(self.contrast_values),
            'max': np.max(self.contrast_values)
        }
        
        # Neue erweiterte Statistiken
        saturation_stats = {
            'mean': np.mean(self.saturation_values) if self.saturation_values else 0.5,
            'std': np.std(self.saturation_values) if self.saturation_values else 0.1
        }
        
        color_temp_stats = {
            'mean': np.mean(self.color_temperature_values) if self.color_temperature_values else 1.0,
            'std': np.std(self.color_temperature_values) if self.color_temperature_values else 0.1
        }
        
        hue_stats = {
            'mean': np.mean(self.hue_values) if self.hue_values else 0.0,
            'std': np.std(self.hue_values) if self.hue_values else 5.0
        }
        
        # Optimierte Parameter-Berechnung
        
        # 1. Helligkeit: Intelligente Berechnung basierend auf Camera-Helligkeit
        avg_brightness = brightness_stats['mean']
        if avg_brightness > 0.7:
            # Helle Camera -> mehr abdunkeln
            brightness_min, brightness_max = 0.7, 1.1
        elif avg_brightness < 0.4:
            # Dunkle Camera -> mehr aufhellen  
            brightness_min, brightness_max = 0.9, 1.3
        else:
            # Normale Camera -> ausgewogen
            brightness_min, brightness_max = 0.8, 1.2
            
        # 2. Kontrast: Basierend auf Camera-Kontrast-Variationen
        contrast_range = contrast_stats['std'] * 1.5
        contrast_min = max(0.85, 1.0 - contrast_range)
        contrast_max = min(1.15, 1.0 + contrast_range)
        
        # 3. Sättigung: Basierend auf Camera-Farbsättigung
        saturation_variation = saturation_stats['std'] * 2.0
        saturation_min = max(0.7, 1.0 - saturation_variation)
        saturation_max = min(1.3, 1.0 + saturation_variation)
        
        # 4. Farbtemperatur: Basierend auf Camera-Farbtemperatur
        color_temp_variation = color_temp_stats['std'] * 1.5
        color_temp_min = max(0.8, 1.0 - color_temp_variation)
        color_temp_max = min(1.2, 1.0 + color_temp_variation)
        
        # 5. Farbton: Basierend auf dominanten Farbtönen
        hue_range = max(5.0, hue_stats['std'] * 2.0)  # Mindestens ±5°
        hue_shift_max = min(15.0, hue_range)  # Maximal ±15°
        
        # 6. Andere Parameter (wie vorher)
        blur_max = max(1.0, np.percentile(self.blur_values, 95))
        noise_max = max(5.0, np.percentile(self.noise_values, 95))
        rotation_max = max(0.5, np.percentile(self.rotation_values, 95))
        shadow_max = max(0.1, np.percentile(self.shadow_values, 95))
        
        optimal_params = {
            # Basis-Parameter (optimiert)
            'brightness_min': round(brightness_min, 2),
            'brightness_max': round(brightness_max, 2),
            'contrast_min': round(contrast_min, 2),
            'contrast_max': round(contrast_max, 2),
            'blur_max': round(blur_max, 1),
            'noise_max': round(noise_max, 1),
            'rotation_max': round(min(rotation_max, 5.0), 1),  # Max 5° für Stabilität
            'shadow': round(shadow_max, 2),
            
            # Neue erweiterte Parameter
            'saturation_min': round(saturation_min, 2),
            'saturation_max': round(saturation_max, 2),
            'color_temperature_min': round(color_temp_min, 2),
            'color_temperature_max': round(color_temp_max, 2),
            'hue_shift_max': round(hue_shift_max, 1)
        }
        
        extended_stats = {
            'brightness': brightness_stats,
            'contrast': contrast_stats,
            'blur': {'mean': np.mean(self.blur_values), 'std': np.std(self.blur_values)},
            'noise': {'mean': np.mean(self.noise_values), 'std': np.std(self.noise_values)},
            'rotation': {'mean': np.mean(self.rotation_values), 'std': np.std(self.rotation_values)},
            'shadow': {'mean': np.mean(self.shadow_values), 'std': np.std(self.shadow_values)},
            'saturation': saturation_stats,
            'color_temperature': color_temp_stats,
            'hue': hue_stats
        }
        
        return optimal_params, extended_stats
    
    def create_visualization(self, output_dir="./analysis_output"):
        """Erstellt Visualisierungen der Analyseergebnisse"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.brightness_values:
            print("❌ Keine Daten für Visualisierung verfügbar!")
            return
        
        # 2x3 Subplot für alle Parameter
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('📷 Pi Camera Image Analysis Results', fontsize=16, fontweight='bold')
        
        # Brightness
        axes[0,0].hist(self.brightness_values, bins=20, alpha=0.7, color='orange')
        axes[0,0].set_title('💡 Brightness Distribution')
        axes[0,0].set_xlabel('Brightness (0-1)')
        axes[0,0].axvline(np.mean(self.brightness_values), color='red', linestyle='--', label=f'Mean: {np.mean(self.brightness_values):.3f}')
        axes[0,0].legend()
        
        # Contrast
        axes[0,1].hist(self.contrast_values, bins=20, alpha=0.7, color='purple')
        axes[0,1].set_title('🎭 Contrast Distribution')
        axes[0,1].set_xlabel('Contrast (0-1)')
        axes[0,1].axvline(np.mean(self.contrast_values), color='red', linestyle='--', label=f'Mean: {np.mean(self.contrast_values):.3f}')
        axes[0,1].legend()
        
        # Blur
        axes[0,2].hist(self.blur_values, bins=20, alpha=0.7, color='blue')
        axes[0,2].set_title('🌀 Blur Distribution')
        axes[0,2].set_xlabel('Blur Factor')
        axes[0,2].axvline(np.mean(self.blur_values), color='red', linestyle='--', label=f'Mean: {np.mean(self.blur_values):.2f}')
        axes[0,2].legend()
        
        # Noise
        axes[1,0].hist(self.noise_values, bins=20, alpha=0.7, color='green')
        axes[1,0].set_title('📊 Noise Distribution')
        axes[1,0].set_xlabel('Noise Level')
        axes[1,0].axvline(np.mean(self.noise_values), color='red', linestyle='--', label=f'Mean: {np.mean(self.noise_values):.1f}')
        axes[1,0].legend()
        
        # Rotation
        axes[1,1].hist(self.rotation_values, bins=20, alpha=0.7, color='red')
        axes[1,1].set_title('🔄 Rotation Distribution')
        axes[1,1].set_xlabel('Rotation (degrees)')
        axes[1,1].axvline(np.mean(self.rotation_values), color='red', linestyle='--', label=f'Mean: {np.mean(self.rotation_values):.1f}°')
        axes[1,1].legend()
        
        # Shadow
        axes[1,2].hist(self.shadow_values, bins=20, alpha=0.7, color='gray')
        axes[1,2].set_title('🌙 Shadow Variation Distribution')
        axes[1,2].set_xlabel('Shadow Factor')
        axes[1,2].axvline(np.mean(self.shadow_values), color='red', linestyle='--', label=f'Mean: {np.mean(self.shadow_values):.3f}')
        axes[1,2].legend()
        
        plt.tight_layout()
        
        # Speichern
        output_file = output_path / "camera_analysis_results.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualisierung gespeichert: {output_file}")

def update_config_with_optimal_parameters(config_path, optimal_params):
    """Aktualisiert die config.yaml mit optimalen Parametern"""
    try:
        # Config laden
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Augmentierung Parameter aktualisieren
        if 'augmentation' not in config:
            config['augmentation'] = {}
        
        # Nur die Parameter aktualisieren, die wir berechnet haben
        config['augmentation']['num_augmentations'] = 30  # Wie gewünscht
        config['augmentation']['brightness_min'] = optimal_params['brightness_min']
        config['augmentation']['brightness_max'] = optimal_params['brightness_max']
        config['augmentation']['contrast_min'] = optimal_params['contrast_min']
        config['augmentation']['contrast_max'] = optimal_params['contrast_max']
        config['augmentation']['blur_max'] = optimal_params['blur_max']
        config['augmentation']['noise_max'] = optimal_params['noise_max']
        config['augmentation']['rotation_max'] = optimal_params['rotation_max']
        config['augmentation']['shadow'] = optimal_params['shadow']
        
        # Backup der alten Config erstellen
        backup_path = f"{config_path}.backup"
        import shutil
        shutil.copy2(config_path, backup_path)
        print(f"💾 Backup erstellt: {backup_path}")
        
        # Neue Config speichern
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Config aktualisiert: {config_path}")
        
    except Exception as e:
        print(f"❌ Fehler beim Aktualisieren der Config: {e}")

def main():
    """Hauptfunktion"""
    # Prüfe ob Play-Button Modus (keine sys.argv Parameter)
    if len(sys.argv) == 1:
        print("🎮 Play-Button Modus: Nutze Standard-Parameter aus config.yaml")
        try:
            config = load_config()
        except Exception as e:
            print(f"❌ Fehler beim Laden der Config: {e}")
            config = {'data': {'camera_images': './data/camera_images'}}
    else:
        # Argument Parser für Terminal-Nutzung
        parser = argparse.ArgumentParser(description='Analysiert Camera-Bilder für optimale Augmentierung')
        parser.add_argument('--camera-dir', 
                          help='Pfad zum Camera-Bilder Verzeichnis', 
                          default='./data/camera_images')
        parser.add_argument('--config', 
                          help='Pfad zur config.yaml', 
                          default='config.yaml')
        parser.add_argument('--output-dir',
                          help='Ausgabeverzeichnis für Analyse-Ergebnisse',
                          default='./analysis_output')
        parser.add_argument('--update-config',
                          action='store_true',
                          help='Aktualisiere config.yaml automatisch mit optimalen Parametern')
        
        args = parser.parse_args()
        config = {'data': {'camera_images': args.camera_dir}}
    
    print("🔬 MTG Camera Image Analyzer")
    print("=" * 50)
    print()
    
    # Analyzer initialisieren
    analyzer = CameraImageAnalyzer()
    
    # Camera-Verzeichnis aus Config
    camera_dir = config['data']['camera_images']
    
    # Analyse durchführen
    analyzer.analyze_directory(camera_dir)
    
    # Optimale Parameter berechnen
    result = analyzer.calculate_optimal_parameters()
    if result is None:
        print("❌ Keine Analyseergebnisse verfügbar!")
        return
    
    optimal_params, stats = result
    
    # Ergebnisse anzeigen
    print()
    print("🎯 OPTIMALE AUGMENTIERUNGSPARAMETER")
    print("=" * 50)
    print(f"📊 Basierend auf {len(analyzer.brightness_values)} analysierten Bildern")
    print()
    
    print("📋 Empfohlene config.yaml Parameter (ERWEITERT):")
    print("```yaml")
    print("augmentation:")
    print(f"  num_augmentations: 30  # Erhöht für 512D Training")
    
    # Basis-Parameter
    print(f"  brightness_min: {optimal_params['brightness_min']}")
    print(f"  brightness_max: {optimal_params['brightness_max']}")
    print(f"  contrast_min: {optimal_params['contrast_min']}")
    print(f"  contrast_max: {optimal_params['contrast_max']}")
    print(f"  blur_max: {optimal_params['blur_max']}")
    print(f"  noise_max: {optimal_params['noise_max']}")
    print(f"  rotation_max: {optimal_params['rotation_max']}")
    print(f"  shadow: {optimal_params['shadow']}")
    
    # Neue erweiterte Parameter
    print("  # Neue Farbparameter:")
    print(f"  saturation_min: {optimal_params['saturation_min']}")
    print(f"  saturation_max: {optimal_params['saturation_max']}")
    print(f"  color_temperature_min: {optimal_params['color_temperature_min']}")
    print(f"  color_temperature_max: {optimal_params['color_temperature_max']}")
    print(f"  hue_shift_max: {optimal_params['hue_shift_max']}")
    
    print("  perspective: 0.05  # Unverändert")
    print("  background_color: \"white\"  # Unverändert")
    print("```")
    print()
    
    # Erweiterte Statistiken anzeigen
    print("📈 ERWEITERTE ANALYSE-STATISTIKEN")
    print("=" * 40)
    
    # Basis-Parameter
    print("🔧 BASIS-PARAMETER:")
    for param_name in ['brightness', 'contrast', 'blur', 'noise', 'rotation', 'shadow']:
        if param_name in stats:
            param_stats = stats[param_name]
            print(f"  {param_name.upper()}: Ø {param_stats['mean']:.3f} ± {param_stats['std']:.3f}")
    
    print()
    print("🎨 NEUE FARB-PARAMETER:")
    
    # Sättigung
    if 'saturation' in stats:
        sat_stats = stats['saturation']
        print(f"  SÄTTIGUNG: Ø {sat_stats['mean']:.3f} ± {sat_stats['std']:.3f}")
        if sat_stats['mean'] > 0.6:
            print(f"    → Kräftige Farben in Camera-Bildern")
        elif sat_stats['mean'] < 0.3:
            print(f"    → Eher schwache Farben in Camera-Bildern")
        else:
            print(f"    → Ausgewogene Farbsättigung")
    
    # Farbtemperatur
    if 'color_temperature' in stats:
        temp_stats = stats['color_temperature']
        print(f"  FARBTEMPERATUR: Ø {temp_stats['mean']:.3f} ± {temp_stats['std']:.3f}")
        if temp_stats['mean'] > 1.1:
            print(f"    → Camera tendiert zu warmen Tönen (gelblich/rötlich)")
        elif temp_stats['mean'] < 0.9:
            print(f"    → Camera tendiert zu kalten Tönen (bläulich)")
        else:
            print(f"    → Ausgeglichene Farbtemperatur")
    
    # Farbton
    if 'hue' in stats:
        hue_stats = stats['hue']
        print(f"  FARBTON-VERSCHIEBUNG: Ø {hue_stats['mean']:+.1f}° ± {hue_stats['std']:.1f}°")
        if abs(hue_stats['mean']) > 5:
            print(f"    → Systematische Farbverschiebung erkannt")
        else:
            print(f"    → Keine systematische Farbverschiebung")
    
    print()
    # Visualisierung erstellen
    if len(sys.argv) == 1:  # Play-Button Modus
        analyzer.create_visualization()
    else:
        analyzer.create_visualization(args.output_dir)
    
    # Nur Empfehlungen anzeigen, NICHT automatisch config.yaml überschreiben
    print()
    print("📋 EMPFOHLENE CONFIG.YAML PARAMETER:")
    print("=" * 50)
    print("Kopiere diese Werte manuell in deine config.yaml:")
    print()
    print("augmentation:")
    print(f"  num_augmentations: 30")
    for key, value in optimal_params.items():
        print(f"  {key}: {value}")
    print()
    
    if len(sys.argv) > 1 and hasattr(args, 'update_config') and args.update_config:
        # Nur wenn explizit --update-config verwendet wird
        config_path = args.config
        update_config_with_optimal_parameters(config_path, optimal_params)
        print("✅ Config-Datei wurde aktualisiert!")
    else:
        print("💡 Hinweis: Verwende --update-config um die config.yaml automatisch zu aktualisieren")
    
    print()
    print("🚀 NÄCHSTE SCHRITTE:")
    print("1. Überprüfe die empfohlenen Parameter oben")
    print("2. Trage sie manuell in config.yaml ein")
    print("3. Führe aus: python src/cardscanner/augment_cards.py")
    print("4. Trainiere neu: python src/cardscanner/train_triplet.py")
    print("5. Erstelle neue Embeddings: python src/cardscanner/generate_embeddings.py")
    
    print()
    print("✅ Camera Image Analyse abgeschlossen!")

if __name__ == "__main__":
    main()