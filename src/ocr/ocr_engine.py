"""OCR Engine für Magic-Karten mit festen Crop-Koordinaten."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

# Projekt-Root zum Path hinzufügen
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.ocr_result import OcrResult

# Tesseract-Pfad (Windows)
# Falls Tesseract nicht im PATH ist, hier den vollständigen Pfad angeben:
try:
    import pytesseract
    # Versuche Standardpfad
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception:
    pass  # Wird später beim Import geprüft

# Fallback Pixel-Konstanten (werden überschrieben falls config ocr_crops vorhanden)
CROP_X_START_FALLBACK = 200
CROP_X_END_FALLBACK = 2850
CROP_START_Y_FALLBACK = 500
CROP_HEIGHT_FALLBACK = 800
COLLECTOR_NUMBER_CROP_FALLBACK = (100, 2700, 500, 80)
SETID_CROP_FALLBACK = (100, 2800, 500, 80)

# Debug-Verzeichnis
DEBUG_DIR = PROJECT_ROOT / "debug" / "ocr"


def _resolve_crops_from_config(config: dict) -> dict:
    """Extrahiert Crop-Werte aus config (ocr_crops) oder nutzt Fallbacks.

    Returns dict mit keys: name, collector, setid.
    """
    ocr_cfg = (config or {}).get("ocr_crops", {})
    name_cfg = ocr_cfg.get("name", {})
    collector_cfg = ocr_cfg.get("collector", {})
    setid_cfg = ocr_cfg.get("setid", {})
    # Name unterstützt neues Schema (x,y,w,h) und altes (x_start,x_end,height)
    if "x" in name_cfg and "w" in name_cfg:
        name = {
            "x_start": int(name_cfg.get("x", CROP_X_START_FALLBACK)),
            "x_end": int(name_cfg.get("x", CROP_X_START_FALLBACK)) + int(name_cfg.get("w", CROP_X_END_FALLBACK - CROP_X_START_FALLBACK)),
            "y": int(name_cfg.get("y", CROP_START_Y_FALLBACK)),
            "height": int(name_cfg.get("h", CROP_HEIGHT_FALLBACK)),
        }
    else:
        name = {
            "x_start": int(name_cfg.get("x_start", CROP_X_START_FALLBACK)),
            "x_end": int(name_cfg.get("x_end", CROP_X_END_FALLBACK)),
            "y": int(name_cfg.get("y", CROP_START_Y_FALLBACK)),
            "height": int(name_cfg.get("height", CROP_HEIGHT_FALLBACK)),
        }
    collector = {
        "x": int(collector_cfg.get("x", COLLECTOR_NUMBER_CROP_FALLBACK[0])),
        "y": int(collector_cfg.get("y", COLLECTOR_NUMBER_CROP_FALLBACK[1])),
        "w": int(collector_cfg.get("w", COLLECTOR_NUMBER_CROP_FALLBACK[2])),
        "h": int(collector_cfg.get("h", COLLECTOR_NUMBER_CROP_FALLBACK[3])),
    }
    setid = {
        "x": int(setid_cfg.get("x", SETID_CROP_FALLBACK[0])),
        "y": int(setid_cfg.get("y", SETID_CROP_FALLBACK[1])),
        "w": int(setid_cfg.get("w", SETID_CROP_FALLBACK[2])),
        "h": int(setid_cfg.get("h", SETID_CROP_FALLBACK[3])),
    }
    return {"name": name, "collector": collector, "setid": setid}


def _crop_name_region(image: Image.Image, name_cfg: dict) -> Image.Image:
    width, height = image.size
    left = max(0, min(name_cfg["x_start"], width - 1))
    right = max(left + 1, min(name_cfg["x_end"], width))
    top = max(0, min(name_cfg["y"], height - 1))
    bottom = max(top + 1, min(name_cfg["y"] + name_cfg["height"], height))
    return image.crop((left, top, right, bottom))


def _crop_collector_number(image: Image.Image, collector_cfg: tuple | dict) -> Image.Image:
    """
    Erzeugt Collector-Number-Crop mit festen Koordinaten.
    
    Args:
        image: PIL Image der Karte
        
    Returns:
        Gecroptes Bild mit Collector Number
    """
    if isinstance(collector_cfg, dict):
        x, y, w, h = collector_cfg["x"], collector_cfg["y"], collector_cfg["w"], collector_cfg["h"]
    else:
        x, y, w, h = collector_cfg
    return image.crop((x, y, x + w, y + h))


def _crop_setid(image: Image.Image, setid_cfg: tuple | dict) -> Image.Image:
    """
    Erzeugt Set-ID-Crop mit festen Koordinaten.
    
    Args:
        image: PIL Image der Karte
        
    Returns:
        Gecroptes Bild mit Set-ID
    """
    if isinstance(setid_cfg, dict):
        x, y, w, h = setid_cfg["x"], setid_cfg["y"], setid_cfg["w"], setid_cfg["h"]
    else:
        x, y, w, h = setid_cfg
    return image.crop((x, y, x + w, y + h))


def _save_debug_crops(
    card_image_path: str,
    name_crop: Image.Image,
    collector_crop: Image.Image,
    setid_crop: Image.Image,
) -> None:
    """
    Speichert Debug-Crops unter debug/ocr/.
    
    Args:
        card_image_path: Pfad zum Original-Bild
        name_crop: Name-Crop
        collector_crop: Collector-Number-Crop
        setid_crop: Set-ID-Crop
    """
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(card_image_path).stem
        
        name_crop.save(DEBUG_DIR / f"{base_name}_name.png")
        collector_crop.save(DEBUG_DIR / f"{base_name}_collector.png")
        setid_crop.save(DEBUG_DIR / f"{base_name}_setid.png")
    except Exception as e:
        print(f"[WARN] Konnte Debug-Crops nicht speichern: {e}")


def run_ocr_for_card_image(
    card_image_path: str,
    config: dict,
) -> Optional[OcrResult]:
    """
    Führt OCR auf einem Kartenbild aus.
    
    Workflow:
    1. Bild laden
    2. Name-Crop mit festen Koordinaten
    3. Collector-Number-Crop
    4. Set-ID-Crop
    5. OCR auf allen Crops
    6. Texte bereinigen
    7. Best-Name ermitteln
    8. OCR-Qualität bewerten
    9. OcrResult zurückgeben
    
    Args:
        card_image_path: Pfad zum Kamera-Bild
        config: Konfiguration (für zukünftige Erweiterungen)
        
    Returns:
        OcrResult oder None bei Fehler
    """
    try:
        # Prüfe ob ocr.py Funktionen verfügbar sind
        try:
            from src.ocr.ocr import (
                find_best_rapidfuzz,
                evaluate_collector_setid_quality,
            )
            from src.ocr.regex import (
                clean_card_name,
                clean_collector_number,
                clean_set_id,
            )
            import pytesseract
        except ImportError as e:
            print(f"[ERROR] OCR-Funktionen nicht gefunden: {e}")
            return None
        
        # 1. Bild laden
        if not os.path.exists(card_image_path):
            print(f"[ERROR] Bild nicht gefunden: {card_image_path}")
            return None
        
        image = Image.open(card_image_path).convert("RGB")
        
        # 2-4. Crops erzeugen (aus config oder Fallback)
        crop_cfg = _resolve_crops_from_config(config)
        name_crop = _crop_name_region(image, crop_cfg["name"])
        collector_crop = _crop_collector_number(image, crop_cfg["collector"])
        setid_crop = _crop_setid(image, crop_cfg["setid"])

        print(f"[OCR-CROPS] name={crop_cfg['name']} collector={crop_cfg['collector']} setid={crop_cfg['setid']}")
        
        # Debug-Crops speichern
        _save_debug_crops(card_image_path, name_crop, collector_crop, setid_crop)
        
        # 5. OCR ausführen (pytesseract direkt)
        name_raw = pytesseract.image_to_string(name_crop, lang="eng", config="--psm 7")
        collector_raw = pytesseract.image_to_string(collector_crop, lang="eng", config="--psm 7")
        setid_raw = pytesseract.image_to_string(setid_crop, lang="eng", config="--psm 7")
        
        # 6. Texte bereinigen
        name_clean = clean_card_name(name_raw)
        collector_clean = clean_collector_number(collector_raw)
        setid_clean = clean_set_id(setid_raw)
        
        # 7. Best-Name ermitteln (mehrere Kandidaten möglich)
        name_candidates = [name_clean] if name_clean else []
        best_name = find_best_rapidfuzz(name_candidates) if name_candidates else ""
        
        # 8. OCR-Qualität bewerten (kombinierter Text für Scoring)
        combined_text = f"{collector_raw}\n{setid_raw}"
        collector_set_score = evaluate_collector_setid_quality(combined_text)
        
        # 9. OcrResult erstellen
        result = OcrResult(
            best_name=best_name,
            name_candidates=name_candidates,
            collector_raw=collector_raw,
            setid_raw=setid_raw,
            collector_clean=collector_clean,
            setid_clean=setid_clean,
            collector_set_score=collector_set_score,
        )
        
        print(f"[OCR] Name: {best_name}")
        print(f"[OCR] Collector: {collector_clean} (raw: {collector_raw})")
        print(f"[OCR] Set-ID: {setid_clean} (raw: {setid_raw})")
        print(f"[OCR] Quality Score: {collector_set_score}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] OCR fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return None
