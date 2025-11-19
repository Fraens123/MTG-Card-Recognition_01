"""OCR Result Dataclass für Magic-Karten."""

from dataclasses import dataclass
from typing import List


@dataclass
class OcrResult:
    """Ergebnis der OCR-Erkennung auf einer Magic-Karte."""
    
    best_name: str
    name_candidates: List[str]
    collector_raw: str
    setid_raw: str
    collector_clean: str
    setid_clean: str
    collector_set_score: int
