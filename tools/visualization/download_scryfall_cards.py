#!/usr/bin/env python3
"""
Script zum Herunterladen von MTG-Karten von Scryfall für das Training.
"""

import os
import requests
import json
from pathlib import Path
from urllib.parse import urlparse
import time


def download_image(url: str, filepath: str) -> bool:
    """Lädt ein Bild von einer URL herunter"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"❌ Fehler beim Herunterladen {url}: {e}")
        return False


def search_card_scryfall(card_name: str) -> dict:
    """Sucht eine Karte auf Scryfall"""
    try:
        # Zuerst mit exaktem Namen versuchen
        url = f"https://api.scryfall.com/cards/named?exact={card_name}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 404:
            # Falls nicht gefunden, mit Fuzzy Search versuchen
            url = f"https://api.scryfall.com/cards/named?fuzzy={card_name}"
            response = requests.get(url, timeout=10)
        
        response.raise_for_status()
        return response.json()
    
    except Exception as e:
        print(f"❌ Karte '{card_name}' nicht gefunden: {e}")
        return None


def download_scryfall_card(card_name: str, output_dir: str) -> bool:
    """Lädt eine einzelne Karte von Scryfall herunter"""
    print(f"🔍 Suche Karte: {card_name}")
    
    card_data = search_card_scryfall(card_name)
    if not card_data:
        return False
    
    # Extrahiere Karteninformationen
    card_uuid = card_data.get('id')
    set_code = card_data.get('set', 'UNK').upper()
    collector_number = card_data.get('collector_number', '000')
    name = card_data.get('name', card_name).replace(' ', '_').replace(',', '').replace("'", '')
    
    # Image URL ermitteln (bevorzuge normal, falls nicht vorhanden dann large)
    image_uris = card_data.get('image_uris', {})
    if not image_uris and 'card_faces' in card_data:
        # Doppelseitige Karten - nimm die erste Seite
        image_uris = card_data['card_faces'][0].get('image_uris', {})
    
    image_url = image_uris.get('normal') or image_uris.get('large')
    if not image_url:
        print(f"❌ Kein Kartenbild gefunden für {card_name}")
        return False
    
    # Dateiname erstellen: {set}_{number}_{name}_{uuid}.jpg
    filename = f"{set_code}_{collector_number}_{name}_{card_uuid}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # Bild herunterladen
    print(f"⬇️  Lade herunter: {filename}")
    success = download_image(image_url, filepath)
    
    if success:
        print(f"✅ Erfolgreich: {filename}")
        return True
    else:
        return False


def get_card_names_from_camera_images(camera_dir: str) -> list:
    """Extrahiert Kartennamen aus camera_images Ordner"""
    card_names = []
    camera_path = Path(camera_dir)
    
    if not camera_path.exists():
        print(f"❌ Camera-Ordner nicht gefunden: {camera_dir}")
        return card_names
    
    for image_file in camera_path.glob("*.jpg"):
        # Extrahiere Kartenname (entferne _01.jpg am Ende)
        name_part = image_file.stem
        if name_part.endswith('_01'):
            name_part = name_part[:-3]
        
        # Ersetze Unterstriche durch Leerzeichen für Scryfall-Suche
        card_name = name_part.replace('_', ' ')
        card_names.append(card_name)
    
    return card_names


def main():
    """Hauptfunktion"""
    # Ausgabeordner erstellen
    output_dir = "data/scryfall_images"
    camera_dir = "data/camera_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Kartennamen aus camera_images extrahieren
    card_names = get_card_names_from_camera_images(camera_dir)
    
    if not card_names:
        print("❌ Keine Karten im camera_images Ordner gefunden!")
        return
    
    print(f"🚀 Lade {len(card_names)} Karten von Scryfall herunter...")
    print(f"📁 Quelle: {camera_dir}")
    print(f"📁 Ziel: {output_dir}")
    print(f"🃏 Gefundene Karten: {', '.join(card_names)}")
    
    successful = 0
    
    for i, card_name in enumerate(card_names, 1):
        print(f"\n[{i}/{len(card_names)}] {card_name}")
        
        if download_scryfall_card(card_name, output_dir):
            successful += 1
        
        # Kurze Pause zwischen Downloads (Scryfall Rate Limiting)
        if i < len(card_names):
            time.sleep(1)
    
    print(f"\n📊 Download abgeschlossen!")
    print(f"✅ Erfolgreich: {successful}/{len(card_names)} Karten")
    
    if successful > 0:
        print(f"📁 Scryfall-Bilder gespeichert in: {output_dir}")
    else:
        print("❌ Keine Karten heruntergeladen")


if __name__ == "__main__":
    main()