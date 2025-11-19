"""OCR-Module für Magic-Karten-Erkennung."""

from .ocr_result import OcrResult
from .ocr_engine import run_ocr_for_card_image
from .ocr_scoring import score_candidate_with_ocr, select_print_with_ocr

__all__ = [
    "OcrResult",
    "run_ocr_for_card_image",
    "score_candidate_with_ocr",
    "select_print_with_ocr",
]
