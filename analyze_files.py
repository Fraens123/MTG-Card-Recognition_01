#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from cardscanner.dataset import parse_scryfall_filename

def main():
    images_dir = project_root / "data" / "scryfall_augmented"
    
    # Sammle alle Dateien
    image_files = list(images_dir.glob("*.jpg"))
    print(f"Gefundene Bilder: {len(image_files)}")
    
    # Parse alle Dateien
    unique_cards = set()
    unique_original_cards = set()  # Nur ohne Augmentierung
    sets_found = set()
    
    for image_file in image_files:
        try:
            # Clean filename for augmented images
            clean_name = image_file.name
            if "_aug_" in clean_name:
                clean_name = clean_name.split("_aug_")[0] + ".jpg"
            elif "_original" in clean_name:
                clean_name = clean_name.replace("_original", "")
            
            card_info = parse_scryfall_filename(clean_name)
            if card_info:
                card_uuid, set_code, collector_number, name = card_info
                
                # Erstelle eindeutigen Identifier
                unique_id = f"{set_code}_{collector_number}_{name}"
                unique_cards.add(unique_id)
                
                # Nur für Original-Bilder (ohne Augmentierung)
                if "_original" in image_file.name or ("_aug_" not in image_file.name):
                    unique_original_cards.add(unique_id)
                
                sets_found.add(set_code)
        except Exception as e:
            print(f"Fehler bei {image_file.name}: {e}")
    
    print(f"\nGefundene Sets: {len(sets_found)}")
    print(f"Alle eindeutigen Set+Number+Name Kombinationen: {len(unique_cards)}")
    print(f"Einzigartige Karten (ohne Augmentierung): {len(unique_original_cards)}")
    
    # Zeige erste 10 Sets
    print(f"\nErste 10 Sets:")
    for i, set_code in enumerate(sorted(sets_found)[:10]):
        print(f"  {i+1:2d}. {set_code}")
    
    if len(sets_found) > 10:
        print(f"  ... und {len(sets_found)-10} weitere Sets")

if __name__ == "__main__":
    main()