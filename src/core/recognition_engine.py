"""
Recognition Engine für MTG-Karten mit Scryfall-ID + Oracle-ID Grouping

Datenfluss:
1. Bild → CNN → Query-Embedding
2. Vektorsuche findet Top-k Prints (Scryfall-IDs)
3. Prints nach oracle_id gruppieren
4. Beste oracle_id-Gruppe wählen
5. Innerhalb Gruppe besten Print mit Sprach-Präferenz wählen
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from src.core.embedding_utils import build_card_embedding
from src.core.image_ops import crop_card_art
from src.core.sqlite_store import load_embeddings_with_meta


# Konfiguration: Bevorzugte Sprachen (höhere Priorität zuerst)
PREFERRED_LANGS = ["de", "en"]


@dataclass
class CardMeta:
    """Metadaten einer Karte (Print)."""
    scryfall_id: str
    oracle_id: str
    name: str
    set_code: str
    collector_number: str
    lang: Optional[str]
    image_path: str


@dataclass
class PrintMatch:
    """Ein einzelner Print-Match mit Similarity."""
    scryfall_id: str
    oracle_id: str
    similarity: float
    distance: float
    meta: CardMeta


@dataclass
class OracleGroup:
    """Gruppe von Prints mit derselben oracle_id."""
    oracle_id: str
    best_similarity: float
    mean_similarity: float
    prints: List[PrintMatch]


@dataclass
class RecognitionResult:
    """Finales Erkennungsergebnis."""
    scryfall_id: str  # Der gewählte Print
    oracle_id: str    # Die logische Karte
    similarity: float
    meta: CardMeta
    oracle_group: OracleGroup
    top_k_prints: List[PrintMatch]
    top_oracle_groups: List[OracleGroup]


class EmbeddingIndex:
    """Index für Print-Embeddings (Scryfall-ID basiert)."""
    
    def __init__(self, db_path: str, mode: str = "runtime", emb_dim: int = 1024):
        """
        Lädt Embeddings aus SQLite, gruppiert nach scryfall_id.
        
        Args:
            db_path: Pfad zur SQLite-Datenbank
            mode: Embedding-Modus ("runtime" oder "analysis")
            emb_dim: Embedding-Dimension
        """
        self.db_path = db_path
        self.mode = mode
        self.emb_dim = emb_dim
        
        # Schlüssel = scryfall_id (Print-ID)
        self.embeddings: Dict[str, List[np.ndarray]] = {}
        self.meta: Dict[str, CardMeta] = {}
        
        # Für schnelle Vektorsuche: Flache Arrays
        self.embedding_matrix: Optional[np.ndarray] = None
        self.scryfall_ids: List[str] = []
        
        self._load_from_db()
    
    def _load_from_db(self) -> None:
        """Lädt Embeddings aus der Datenbank."""
        # load_embeddings_with_meta gruppiert nach scryfall_id!
        embeddings_by_card, meta_by_card = load_embeddings_with_meta(
            self.db_path, self.mode, self.emb_dim
        )
        
        # Konvertiere zu CardMeta-Objekten
        for scryfall_id, vecs in embeddings_by_card.items():
            self.embeddings[scryfall_id] = vecs
            
            meta_dict = meta_by_card.get(scryfall_id, {})
            self.meta[scryfall_id] = CardMeta(
                scryfall_id=scryfall_id,
                oracle_id=meta_dict.get("oracle_id", scryfall_id),
                name=meta_dict.get("name", scryfall_id),
                set_code=meta_dict.get("set", ""),
                collector_number=meta_dict.get("collector_number", ""),
                lang=meta_dict.get("lang"),
                image_path=(meta_dict.get("image_paths") or [""])[0],
            )
        
        # Baue flache Matrix für schnelle Suche
        self._build_flat_index()
    
    def _build_flat_index(self) -> None:
        """Baut flachen Index für Vektorsuche."""
        vectors: List[np.ndarray] = []
        ids: List[str] = []
        
        for scryfall_id, vecs in self.embeddings.items():
            for vec in vecs:
                vectors.append(vec.flatten())
                ids.append(scryfall_id)
        
        if vectors:
            self.embedding_matrix = np.stack(vectors, axis=0).astype(np.float32)
            # L2-Normalisierung für Cosine Similarity
            norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
            self.embedding_matrix = self.embedding_matrix / np.clip(norms, 1e-12, None)
        else:
            self.embedding_matrix = None
        
        self.scryfall_ids = ids
    
    def size(self) -> int:
        """Anzahl der Embeddings im Index."""
        return len(self.scryfall_ids) if self.scryfall_ids else 0


def encode_image_to_embedding(
    image: Image.Image,
    model: torch.nn.Module,
    transform: T.Compose,
    crop_cfg: Optional[Dict],
    device: torch.device,
) -> np.ndarray:
    """
    Kodiert ein Bild zu einem normalisierten Embedding-Vektor.
    
    Args:
        image: PIL Image
        model: CNN Encoder-Modell
        transform: Preprocessing-Transform
        crop_cfg: Konfiguration für Artwork-Cropping
        device: torch.device
    
    Returns:
        Normalisierter Embedding-Vektor als numpy array [emb_dim]
    """
    # Artwork-Crop
    art_img = crop_card_art(image, crop_cfg)
    
    # Preprocessing
    tensor = transform(art_img).unsqueeze(0).to(device)
    
    # CNN Forward Pass
    with torch.no_grad():
        embedding = build_card_embedding(model, tensor)
    
    # Als numpy array zurückgeben
    return embedding.cpu().numpy().flatten()


def find_top_k_prints(
    query_vec: np.ndarray,
    index: EmbeddingIndex,
    k: int = 20,
) -> List[PrintMatch]:
    """
    Findet die Top-k ähnlichsten Prints (Scryfall-IDs) zu einem Query-Embedding.
    
    WICHTIG: Sucht auf Print-Ebene (Scryfall-ID), nicht Oracle-ID!
    
    Args:
        query_vec: Query-Embedding [emb_dim]
        index: EmbeddingIndex mit Print-Embeddings
        k: Anzahl der Top-Ergebnisse
    
    Returns:
        Liste von PrintMatch-Objekten, sortiert nach Similarity (absteigend)
    """
    if index.embedding_matrix is None or index.size() == 0:
        return []
    
    # Normalisiere Query
    query = query_vec.flatten().reshape(1, -1).astype(np.float32)
    q_norm = np.linalg.norm(query)
    if q_norm > 0:
        query = query / q_norm
    
    # Cosine Similarity berechnen (bereits normalisierte Vektoren)
    similarities = (query @ index.embedding_matrix.T)[0]
    
    # Top-k Indizes
    top_indices = np.argsort(similarities)[::-1][:k]
    
    # Erstelle PrintMatch-Objekte
    results = []
    for idx in top_indices:
        scryfall_id = index.scryfall_ids[idx]
        meta = index.meta.get(scryfall_id)
        if not meta:
            continue
        
        sim = float(similarities[idx])
        results.append(PrintMatch(
            scryfall_id=scryfall_id,
            oracle_id=meta.oracle_id,
            similarity=sim,
            distance=1.0 - sim,
            meta=meta,
        ))
    
    return results


def aggregate_by_oracle_id(top_k_prints: List[PrintMatch]) -> List[OracleGroup]:
    """
    Gruppiert Top-k Prints nach oracle_id und berechnet Aggregat-Scores.
    
    Args:
        top_k_prints: Liste von PrintMatch-Objekten
    
    Returns:
        Liste von OracleGroup-Objekten, sortiert nach best_similarity (absteigend)
    """
    # Gruppiere nach oracle_id
    groups: Dict[str, List[PrintMatch]] = defaultdict(list)
    for print_match in top_k_prints:
        groups[print_match.oracle_id].append(print_match)
    
    # Erstelle OracleGroup-Objekte
    oracle_groups = []
    for oracle_id, prints in groups.items():
        similarities = [p.similarity for p in prints]
        best_sim = max(similarities)
        mean_sim = sum(similarities) / len(similarities)
        
        oracle_groups.append(OracleGroup(
            oracle_id=oracle_id,
            best_similarity=best_sim,
            mean_similarity=mean_sim,
            prints=prints,
        ))
    
    # Sortiere nach best_similarity absteigend
    oracle_groups.sort(key=lambda g: g.best_similarity, reverse=True)
    
    return oracle_groups


def select_best_print_for_oracle_group(
    oracle_group: OracleGroup,
    preferred_langs: Optional[List[str]] = None,
) -> PrintMatch:
    """
    Wählt den besten Print aus einer Oracle-Gruppe.
    
    Strategie:
    1. Wenn preferred_langs gesetzt: Filtere nach Sprache, wähle besten
    2. Sonst: Wähle Print mit bester Similarity
    
    Args:
        oracle_group: OracleGroup-Objekt
        preferred_langs: Liste bevorzugter Sprachen (z.B. ["de", "en"])
    
    Returns:
        PrintMatch-Objekt des gewählten Prints
    """
    prints = oracle_group.prints
    
    if not prints:
        raise ValueError("Oracle-Gruppe ist leer")
    
    # Strategie 1: Sprach-Präferenz
    if preferred_langs:
        for lang in preferred_langs:
            # Filtere Prints mit dieser Sprache
            lang_prints = [p for p in prints if p.meta.lang == lang]
            if lang_prints:
                # Wähle besten nach Similarity
                return max(lang_prints, key=lambda p: p.similarity)
    
    # Strategie 2: Beste Similarity ohne Sprach-Filter
    return max(prints, key=lambda p: p.similarity)


def recognize_card(
    image: Image.Image,
    model: torch.nn.Module,
    index: EmbeddingIndex,
    transform: T.Compose,
    crop_cfg: Optional[Dict],
    device: torch.device,
    k: int = 20,
    preferred_langs: Optional[List[str]] = None,
) -> Optional[RecognitionResult]:
    """
    Haupt-Erkennungsfunktion.
    
    Ablauf:
    1. image → query_vec (CNN)
    2. Top-k Prints über Vektorsuche (Scryfall-ID-Ebene)
    3. Prints nach oracle_id gruppieren
    4. Beste oracle_id-Gruppe wählen
    5. Innerhalb Gruppe besten Print mit Sprach-Präferenz wählen
    
    Args:
        image: PIL Image
        model: CNN Encoder
        index: EmbeddingIndex mit Print-Embeddings
        transform: Preprocessing-Transform
        crop_cfg: Artwork-Crop-Konfiguration
        device: torch.device
        k: Anzahl Top-Prints für Suche
        preferred_langs: Bevorzugte Sprachen (z.B. ["de", "en"])
    
    Returns:
        RecognitionResult oder None (wenn keine Matches gefunden)
    """
    # Schritt 1: Bild → Embedding
    query_vec = encode_image_to_embedding(image, model, transform, crop_cfg, device)
    
    # Schritt 2: Top-k Prints suchen (Scryfall-ID-Ebene!)
    top_k_prints = find_top_k_prints(query_vec, index, k=k)
    
    if not top_k_prints:
        return None
    
    # Schritt 3: Nach oracle_id gruppieren
    oracle_groups = aggregate_by_oracle_id(top_k_prints)
    
    if not oracle_groups:
        return None
    
    # Schritt 4: Beste oracle_id-Gruppe
    best_group = oracle_groups[0]
    
    # Schritt 5: Besten Print in der Gruppe wählen (mit Sprach-Präferenz)
    best_print = select_best_print_for_oracle_group(best_group, preferred_langs)
    
    # Rückgabe-Struktur
    return RecognitionResult(
        scryfall_id=best_print.scryfall_id,
        oracle_id=best_group.oracle_id,
        similarity=best_print.similarity,
        meta=best_print.meta,
        oracle_group=best_group,
        top_k_prints=top_k_prints,
        top_oracle_groups=oracle_groups,
    )


def debug_print_recognition_result(result: Optional[RecognitionResult]) -> None:
    """
    Gibt Debug-Informationen über das Erkennungsergebnis aus.
    
    Args:
        result: RecognitionResult oder None
    """
    if result is None:
        print("[DEBUG] Keine Karte erkannt")
        return
    
    print("\n" + "=" * 80)
    print("[DEBUG] ERKENNUNGSERGEBNIS")
    print("=" * 80)
    
    # Gewählter Print
    print(f"\n✓ GEWÄHLTER PRINT:")
    print(f"  Scryfall-ID: {result.scryfall_id}")
    print(f"  Oracle-ID:   {result.oracle_id}")
    print(f"  Name:        {result.meta.name}")
    print(f"  Set:         {result.meta.set_code} #{result.meta.collector_number}")
    print(f"  Sprache:     {result.meta.lang or 'unknown'}")
    print(f"  Similarity:  {result.similarity:.4f} ({result.similarity*100:.1f}%)")
    
    # Top-5 Prints
    print(f"\n📊 TOP-5 PRINTS (Scryfall-ID-Ebene):")
    for i, print_match in enumerate(result.top_k_prints[:5], 1):
        print(f"  {i}. {print_match.meta.name} [{print_match.meta.set_code}] "
              f"({print_match.meta.lang}) - sim={print_match.similarity:.4f}")
    
    # Top-3 Oracle-Gruppen
    print(f"\n🔍 TOP-3 ORACLE-GRUPPEN:")
    for i, group in enumerate(result.top_oracle_groups[:3], 1):
        print(f"  {i}. Oracle-ID: {group.oracle_id}")
        print(f"     Best Similarity: {group.best_similarity:.4f}")
        print(f"     Mean Similarity: {group.mean_similarity:.4f}")
        print(f"     Prints in Gruppe: {len(group.prints)}")
        # Zeige erste 2 Prints dieser Gruppe
        for j, p in enumerate(group.prints[:2], 1):
            print(f"       {j}. {p.meta.name} [{p.meta.set_code}] ({p.meta.lang}) - {p.similarity:.4f}")
    
    print("=" * 80 + "\n")
