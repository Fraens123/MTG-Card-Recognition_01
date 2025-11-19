"""OCR-basierte Scoring-Funktionen für Print-Auswahl."""

from __future__ import annotations

from typing import Dict, List

try:
    from rapidfuzz import fuzz
except ImportError:
    print("[WARN] rapidfuzz nicht installiert, OCR-Scoring nicht verfügbar")
    fuzz = None

from src.ocr.ocr_result import OcrResult


def score_candidate_with_ocr(candidate: dict, ocr: OcrResult) -> float:
    """
    Bewertet einen Kandidaten anhand der OCR-Daten.
    
    Vergleicht:
    - OCR-Name ↔ candidate["name"]
    - OCR-Collector ↔ candidate["collector_number"]
    - OCR-SetID ↔ candidate["set_code"]
    
    Args:
        candidate: Dict mit Kartendaten (name, collector_number, set_code)
        ocr: OCR-Ergebnis
        
    Returns:
        Score zwischen 0.0 und 1.0 (höher = besser)
    """
    if fuzz is None:
        # Fallback ohne rapidfuzz
        return 0.0
    
    scores = []
    weights = []
    
    # Name-Vergleich (Gewicht: 3)
    if ocr.best_name and candidate.get("name"):
        name_score = fuzz.ratio(
            ocr.best_name.lower(),
            candidate["name"].lower(),
        ) / 100.0
        scores.append(name_score)
        weights.append(3.0)
    
    # Collector-Number-Vergleich (Gewicht: 2)
    if ocr.collector_clean and candidate.get("collector_number"):
        collector_score = fuzz.ratio(
            ocr.collector_clean.lower(),
            str(candidate["collector_number"]).lower(),
        ) / 100.0
        scores.append(collector_score)
        weights.append(2.0)
    
    # Set-ID-Vergleich (Gewicht: 2)
    if ocr.setid_clean and candidate.get("set_code"):
        setid_score = fuzz.ratio(
            ocr.setid_clean.lower(),
            candidate["set_code"].lower(),
        ) / 100.0
        scores.append(setid_score)
        weights.append(2.0)
    
    # Gewichteter Durchschnitt
    if not scores:
        return 0.0
    
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    total_weight = sum(weights)
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def select_print_with_ocr(
    best_match: dict,
    oracle_candidates: List[dict],
    ocr: OcrResult,
) -> dict:
    """
    Wählt den besten Print aus einer Oracle-Gruppe mittels OCR.
    
    Strategie:
    1. Wenn OCR unbrauchbar (niedriger collector_set_score) → best_match
    2. Sonst: Score für alle Kandidaten berechnen
    3. Kandidat mit höchstem Score zurückgeben
    4. Bei Gleichstand: best_match beibehalten
    
    Args:
        best_match: Der CNN-basierte beste Match
        oracle_candidates: Liste aller Prints mit derselben oracle_id
        ocr: OCR-Ergebnis
        
    Returns:
        Gewählter Print (dict)
    """
    # Qualitäts-Check: Ist OCR überhaupt brauchbar?
    OCR_MIN_QUALITY = 50  # Schwellwert für collector_set_score
    
    if ocr is None or ocr.collector_set_score < OCR_MIN_QUALITY:
        print("[OCR] Qualität zu niedrig, verwende CNN-Match")
        return best_match
    
    if not oracle_candidates:
        print("[OCR] Keine Oracle-Kandidaten, verwende CNN-Match")
        return best_match
    
    # Score für alle Kandidaten berechnen
    scored_candidates = []
    for candidate in oracle_candidates:
        score = score_candidate_with_ocr(candidate, ocr)
        scored_candidates.append((score, candidate))
    
    # Nach Score sortieren (absteigend)
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Besten Kandidaten auswählen
    best_score, best_candidate = scored_candidates[0]
    
    print(f"[OCR] Top-3 Kandidaten:")
    for i, (score, cand) in enumerate(scored_candidates[:3], 1):
        print(f"  {i}. {cand.get('name')} [{cand.get('set_code')}] "
              f"#{cand.get('collector_number')} - Score: {score:.3f}")
    
    # Prüfe ob OCR-basierter Match deutlich besser ist
    CNN_SCORE_THRESHOLD = 0.7  # Mindest-Score für OCR-Override
    
    if best_score < CNN_SCORE_THRESHOLD:
        print(f"[OCR] Score zu niedrig ({best_score:.3f}), verwende CNN-Match")
        return best_match
    
    # Prüfe ob es der CNN-Match ist
    best_match_id = best_match.get("card_uuid") or best_match.get("scryfall_id")
    best_candidate_id = best_candidate.get("card_uuid") or best_candidate.get("scryfall_id")
    
    if best_match_id == best_candidate_id:
        print(f"[OCR] Bestätigt CNN-Match: {best_candidate.get('name')}")
    else:
        print(f"[OCR] Überschreibe CNN-Match → {best_candidate.get('name')} "
              f"[{best_candidate.get('set_code')}] #{best_candidate.get('collector_number')}")
    
    return best_candidate
