import pytesseract
from PIL import Image
import re
from rapidfuzz import fuzz
from .regex import clean_card_name, clean_collector_number, clean_set_id

def extract_text(image_path, lang="eng"):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    return text

def extract_texts_from_crops(input_path, crop_configs, lang="eng"):
    texts = []
    image = Image.open(input_path)
    for output_path, xy, size in crop_configs:
        x, y = xy
        w, h = size
        crop_box = (x, y, x + w, y + h)
        cropped = image.crop(crop_box)
        cropped.save(output_path)
        text = pytesseract.image_to_string(cropped, lang=lang, config="--psm 7")
        texts.append(text.strip())
    return texts

def find_best_rapidfuzz(texts):
    # Vorbereiten: alle Namen bereinigen
    cleaned_texts = [clean_card_name(t) for t in texts]
    # Leere Einträge rausfiltern
    non_empty = [t for t in cleaned_texts if t]
    if not non_empty:
        return ""
    if len(non_empty) == 1:
        return non_empty[0]

    best_text = None
    best_avg_score = 0

    # Für jeden Text den Durchschnitt der Ähnlichkeiten zu allen anderen berechnen
    for i, candidate in enumerate(non_empty):
        total_score = 0
        count = 0
        for j, other in enumerate(non_empty):
            if i != j:
                # fuzz.ratio gibt Wert zwischen 0-100
                score = fuzz.ratio(candidate, other)
                total_score += score
                count += 1
        avg_score = total_score / count if count else 0

        # Bisher bester?
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_text = candidate

    return best_text

def find_best_rapidfuzz_xxxx(texts):
    cleaned_texts = [clean_card_name(t) for t in texts]
    non_empty = [t for t in cleaned_texts if t]
    if not non_empty:
        return ""
    base = non_empty[0]
    if len(non_empty) == 1:
        return base
    best_text = base
    best_score = 0
    for t in non_empty[1:]:
        score = fuzz.ratio(base, t)
        if score > best_score:
            best_score = score
            best_text = t
    return best_text

def print_ocr_results(
    texts_name,
    collector_number_text,
    setid_text,
    clean_card_name,
    clean_collector_number,
    clean_set_id,
    best_text
):
    print("="*40)
    print("Bereinigte Texte:")
    for i, orig in enumerate(texts_name):
        cleaned = clean_card_name(orig)
        score = fuzz.ratio(cleaned, best_text)
        print(f"Original: {orig!r}")
        print(f"Bereinigt: {cleaned!r}")
        print(f"Score zu Best: {score}")
        print("-" * 40)
    print(f"Collector Number Original: {collector_number_text!r}")
    print(f"Collector Number Bereinigt: {clean_collector_number(collector_number_text)!r}")
    print(f"SetID Original: {setid_text!r}")
    print(f"SetID Bereinigt: {clean_set_id(setid_text)!r}")
    print("="*40)
    print("Wahrscheinlichster Kartenname für Skryfall abfrage:")
    print(best_text)
    print("="*40)
    print("Collector Number:")
    print(clean_collector_number(collector_number_text))
    print("SetID:")
    print(clean_set_id(setid_text))
    print("="*40)

def compare_text_pair(text1, text2):
    return fuzz.ratio(text1, text2)

def levenshtein_ratio(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_ratio(s2, s1)
    if len(s2) == 0:
        return 1.0 if len(s1) == 0 else 0.0

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    distance = previous_row[-1]
    max_len = max(len(s1), len(s2))
    return 1 - distance / max_len if max_len > 0 else 1.0

def find_best_levenshtein(texts):
    cleaned_texts = [clean_card_name(t) for t in texts]
    # KEINE print-Ausgabe hier!
    non_empty = [t for t in cleaned_texts if t]
    if not non_empty:
        return ""
    base = non_empty[0]
    if len(non_empty) == 1:
        return base
    best_text = base
    best_score = 0
    for t in non_empty[1:]:
        score = levenshtein_ratio(base, t)
        if score > best_score:
            best_score = score
            best_text = t
    return best_text

def extract_collector_setid_text(image_path, crop_config, lang="eng"):
    """
    Spezialisierte Tesseract-Methode für Collector Number und Set ID
    Optimiert für mehrere Zeilen und Zahlen 0-9
    """
    image = Image.open(image_path)
    output_path, xy, size = crop_config
    x, y = xy
    w, h = size
    crop_box = (x, y, x + w, y + h)
    cropped = image.crop(crop_box)
    cropped.save(output_path)
    
    # Verschiedene Tesseract-Konfigurationen für bessere Erkennung
    # Collector Number Whitelist: Buchstaben, Zahlen, Schrägstrich, Leerzeichen und Punkt
    collector_whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/. "
    
    configs = [
        f"--psm 6 -c tessedit_char_whitelist={collector_whitelist}",  # Mehrere Zeilen, Collector Number Format
        f"--psm 4 -c tessedit_char_whitelist={collector_whitelist}",  # Einzelne Spalte, Collector Number Format
        f"--psm 7 -c tessedit_char_whitelist={collector_whitelist}",  # Einzelne Zeile, Collector Number Format
        f"--psm 8 -c tessedit_char_whitelist={collector_whitelist}",  # Einzelnes Wort, Collector Number Format
        "--psm 6",  # Fallback: Mehrere Zeilen ohne Whitelist
        "--psm 4",  # Fallback: Einzelne Spalte ohne Whitelist
    ]
    
    best_text = ""
    best_confidence = 0
    
    for config in configs:
        try:
            # Versuche Text zu extrahieren
            text = pytesseract.image_to_string(cropped, lang=lang, config=config)
            text = text.strip()
            
            # Bewerte die Qualität des Ergebnisses
            confidence = evaluate_collector_setid_quality(text)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_text = text
                
        except Exception as e:
            print(f"Fehler mit Config {config}: {e}")
            continue
    
    return best_text

def evaluate_collector_setid_quality(text):
    """
    Bewertet die Qualität des erkannten Collector/SetID Textes
    Höhere Werte bedeuten bessere Qualität
    Fokus auf Collector Number in der ersten Zeile
    """
    if not text:
        return 0
    
    # Erste Zeile extrahieren (Collector Number ist immer in der ersten Zeile)
    lines = text.split('\n')
    first_line = lines[0].strip() if lines else text.strip()
    
    score = 0
    
    # Bewertung basierend auf der ersten Zeile (Collector Number)
    collector_patterns = [
        r'^[A-Za-z]\d{4}$',           # C0084
        r'^\d{4}\s*[A-Za-z]$',       # 0123 S
        r'^\d{4}$',                   # 0001
        r'^\d{4}/\d{4}$',            # 0023/0124
        r'^[A-Za-z]\d{3}$',          # C084 (kürzere Variante)
        r'^\d{3}/\d{3}$',            # 023/124 (kürzere Variante)
    ]
    
    # Prüfe Collector Number Patterns in der ersten Zeile
    for pattern in collector_patterns:
        if re.match(pattern, first_line.replace(' ', '')):
            score += 20  # Hoher Bonus für erkannte Collector Number Pattern
            break
    
    # Bewerte Anzahl der Zahlen in der ersten Zeile
    digit_count = sum(1 for c in first_line if c.isdigit())
    score += digit_count * 3  # Höherer Bonus für Zahlen
    
    # Bewerte Anzahl der Buchstaben in der ersten Zeile
    letter_count = sum(1 for c in first_line if c.isalpha())
    score += letter_count * 2
    
    # Bonus für Schrägstrich (häufig in Collector Numbers)
    if '/' in first_line:
        score += 8
    
    # Penalisiere sehr kurze oder sehr lange erste Zeilen
    if 3 <= len(first_line) <= 10:
        score += 5
    elif len(first_line) < 3:
        score -= 5
    elif len(first_line) > 12:
        score -= 8
    
    # Penalisiere Sonderzeichen außer Schrägstrich, Leerzeichen und Punkt
    special_chars = sum(1 for c in first_line if not c.isalnum() and c not in ['/', ' ', '.'])
    score -= special_chars * 3
    
    # Bonus wenn mehrere Zeilen vorhanden sind (Set ID ist in zweiter Zeile)
    if len(lines) > 1:
        second_line = lines[1].strip()
        
        # Entferne Sprachangabe nach Punkt (z.B. "WTH.EN" -> "WTH")
        if '.' in second_line:
            second_line = second_line.split('.')[0]
        
        # Set ID Patterns: meist 3 Zeichen, Buchstaben und Zahlen
        setid_patterns = [
            r'^[A-Za-z]{3}$',           # WTH, DRK
            r'^[A-Za-z]\d{2}$',         # M15, V17
            r'^\d[A-Za-z]{2}$',         # 7ED
            r'^[A-Za-z]{3}/[A-Za-z]{3}$', # TSP/TSB (Sonderfall)
            r'^[A-Za-z]{2}\d$',         # Variante mit 2 Buchstaben + 1 Zahl
            r'^[A-Za-z]\d[A-Za-z]$',    # Variante mit Buchstabe-Zahl-Buchstabe
        ]
        
        # Prüfe Set ID Patterns in der zweiten Zeile
        for pattern in setid_patterns:
            if re.match(pattern, second_line):
                score += 25  # Sehr hoher Bonus für erkannte Set ID Pattern
                break
        else:
            # Fallback: Bewerte allgemein nach Länge und Zeichen
            if 2 <= len(second_line) <= 4:
                score += 10
                # Bonus für Buchstaben in Set ID
                if sum(1 for c in second_line if c.isalpha()) >= 2:
                    score += 5
                # Bonus für Zahlen in Set ID
                if sum(1 for c in second_line if c.isdigit()) >= 1:
                    score += 3
    
    return score