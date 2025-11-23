from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_utils import load_config
from src.core.embedding_utils import l2_normalize
from src.core.sqlite_embeddings import (
    load_embeddings_grouped_by_oracle,
    ensure_oracle_quality_table,
)

# ---------- Common helpers ----------

def _read_oracle_quality(sqlite_path: Path, scenario: str) -> List[Dict]:
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    rows: List[Dict] = []
    try:
        for r in conn.execute(
            """
            SELECT oracle_id,
                   scenario,
                   suspect_overlap,
                   suspect_cluster_spread,
                   suspect_metadata_conflict,
                   nearest_neighbor_oracle_id,
                   nearest_neighbor_dist,
                   intra_mean_dist,
                   intra_max_dist,
                   meta_info
            FROM oracle_quality
            WHERE scenario = ?
            ORDER BY oracle_id
            """,
            (scenario,),
        ):
            d = dict(r)
            try:
                d["meta_info"] = json.loads(d.get("meta_info") or "{}")
            except Exception:
                d["meta_info"] = {}
            rows.append(d)
    finally:
        conn.close()
    return rows


def _read_oracle_flags_map(sqlite_path: Path, scenario: str) -> Dict[str, Dict]:
    # map oracle_id -> row
    rows = _read_oracle_quality(sqlite_path, scenario)
    return {r["oracle_id"]: r for r in rows}


def _thumb_name(path: str) -> str:
    h = hashlib.md5(os.path.abspath(path).encode("utf-8")).hexdigest()
    base = os.path.basename(path)
    root, ext = os.path.splitext(base)
    ext = ext.lower() if ext.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
    return f"{root[:40]}_{h[:12]}{ext}"


def _make_thumbnail(src_path: str, dst_dir: Path, size: int) -> Optional[str]:
    try:
        if not src_path or not os.path.isfile(src_path):
            return None
        dst_dir.mkdir(parents=True, exist_ok=True)
        out_name = _thumb_name(src_path)
        out_path = dst_dir / out_name
        if not out_path.exists():
            with Image.open(src_path) as im:
                im = im.convert("RGB")
                im.thumbnail((size, size))
                im.save(out_path, quality=90)
        return str(out_path)
    except Exception:
        return None

# ---------- Cluster report ----------

def render_cluster_report(
    out_html: Path,
    rows: List[Dict],
    meta: Dict[str, Dict],
    suspects_only: bool,
    max_images: int,
    thumb_size: int,
    title: str,
) -> Path:
    assets = out_html.parent / "html_assets" / "thumbnails"

    def _bool_bad(v: int) -> bool:
        try:
            return int(v) != 0
        except Exception:
            return False

    def _sort_key(row: Dict) -> Tuple:
        ov = float(row.get("nearest_neighbor_dist") or float("inf"))
        im = float(row.get("intra_mean_dist") or 0.0)
        return (
            -int(_bool_bad(row.get("suspect_overlap")) or _bool_bad(row.get("suspect_cluster_spread")) or _bool_bad(row.get("suspect_metadata_conflict"))),
            ov,
            -im,
        )

    def _row_to_html(r: Dict) -> str:
        oid = r["oracle_id"]
        m = meta.get(oid, {})
        nb_id = r.get("nearest_neighbor_oracle_id")
        nb_meta = meta.get(nb_id, {}) if nb_id else {}
        rep_path = m.get("image_path") or (m.get("image_paths_all") or [None])[0]
        nb_rep = (nb_meta.get("image_path") if nb_meta else None) or ((nb_meta.get("image_paths_all") or [None])[0] if nb_meta else None)
        rep_thumb = _make_thumbnail(rep_path, assets, thumb_size) if rep_path else None
        nb_thumb = _make_thumbnail(nb_rep, assets, thumb_size) if nb_rep else None
        rep_img_tag = f'<img src="{html.escape(os.path.relpath(rep_thumb, out_html.parent))}" />' if rep_thumb else "<div class=placeholder>no image</div>"
        nb_img_tag = f'<img src="{html.escape(os.path.relpath(nb_thumb, out_html.parent))}" />' if nb_thumb else "<div class=placeholder>no image</div>"
        name = m.get("name") or oid
        set_code = m.get("set_code") or ""
        set_name = m.get("set_name") or ""
        colors = json.dumps(m.get("colors", []), ensure_ascii=False)
        mana = m.get("mana_cost") or ""
        cmc = m.get("cmc")
        cmc_txt = f"{cmc:.2f}" if isinstance(cmc, (int, float)) else ""
        f_overlap = _bool_bad(r.get("suspect_overlap"))
        f_spread = _bool_bad(r.get("suspect_cluster_spread"))
        f_meta = _bool_bad(r.get("suspect_metadata_conflict"))
        flag_classes = " ".join([
            "flag-overlap" if f_overlap else "",
            "flag-spread" if f_spread else "",
            "flag-meta" if f_meta else "",
        ]).strip()
        gallery_items: List[str] = []
        for p in (m.get("image_paths_all") or [])[:max_images]:
            t = _make_thumbnail(p, assets, thumb_size)
            if t:
                rel = html.escape(os.path.relpath(t, out_html.parent))
                gallery_items.append(f'<img src="{rel}" />')
        gallery_html = "".join(gallery_items)
        neighbor_name = nb_meta.get("name") if nb_meta else ""
        dist = r.get("nearest_neighbor_dist")
        dist_txt = f"{float(dist):.4f}" if isinstance(dist, (int, float)) else ""
        intra_mean = r.get("intra_mean_dist")
        intra_max = r.get("intra_max_dist")
        intra_mean_txt = f"{float(intra_mean):.4f}" if isinstance(intra_mean, (int, float)) else ""
        intra_max_txt = f"{float(intra_max):.4f}" if isinstance(intra_max, (int, float)) else ""
        return f"""
        <section id=\"cluster-{html.escape(oid)}\" class=\"cluster {flag_classes}\">\n          <div class=\"head\">\n            <div class=\"title\">{html.escape(name)}<span class=\"oid\">{html.escape(oid)}</span></div>\n            <div class=\"meta\">\n              <span>Set: {html.escape(set_code)} {html.escape(set_name)}</span>\n              <span>Mana: {html.escape(str(mana))} | CMC: {html.escape(cmc_txt)}</span>\n              <span>Colors: {html.escape(colors)}</span>\n            </div>\n            <div class=\"stats\">\n              <span class=\"stat\">NN: {html.escape(neighbor_name)} <em>({html.escape(nb_id or '')})</em></span>\n              <span class=\"stat\">dist: {html.escape(dist_txt)}</span>\n              <span class=\"stat\">intra μ: {html.escape(intra_mean_txt)}</span>\n              <span class=\"stat\">intra max: {html.escape(intra_max_txt)}</span>\n              <span class=\"flags\">overlap: {int(f_overlap)} | spread: {int(f_spread)} | meta: {int(f_meta)}</span>\n            </div>\n          </div>\n          <div class=\"images\">\n            <div class=\"card\">{rep_img_tag}</div>\n            <div class=\"card neighbor\">{nb_img_tag}</div>\n          </div>\n          <div class=\"gallery\">{gallery_html}</div>\n        </section>\n        """

    # filter + sort
    if suspects_only:
        rows = [r for r in rows if any(int(r.get(k, 0)) for k in ("suspect_overlap","suspect_cluster_spread","suspect_metadata_conflict"))]
    rows = sorted(rows, key=_sort_key)

    sections = ["".join(_row_to_html(r) for r in rows)]

    doc = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #0b0d10; color: #d9e1ea; }}
    header {{ position: sticky; top: 0; z-index: 2; background: #10151b; border-bottom: 1px solid #1d2630; padding: 10px 16px; }}
    header .title {{ font-weight: 600; }}
    header .legend {{ font-size: 12px; color: #9fb0c3; margin-top: 4px; }}
    main {{ padding: 12px 16px 40px; }}

    .cluster {{ border: 1px solid #1d2630; border-radius: 10px; padding: 10px; margin: 12px 0; background: #121820; }}
    .cluster.flag-overlap {{ box-shadow: inset 0 0 0 2px #ff6262; }}
    .cluster.flag-spread {{ box-shadow: inset 0 0 0 2px #ffa94d; }}
    .cluster.flag-meta {{ box-shadow: inset 0 0 0 2px #70d5ff; }}

    .cluster .head {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: baseline; }}
    .cluster .head .title {{ font-size: 18px; font-weight: 600; }}
    .cluster .head .oid {{ margin-left: 12px; font-size: 12px; color: #9fb0c3; }}
    .cluster .head .meta {{ display: flex; gap: 14px; font-size: 12px; color: #9fb0c3; flex-wrap: wrap; }}
    .cluster .head .stats {{ display: flex; gap: 14px; font-size: 12px; color: #cfd8e3; flex-wrap: wrap; }}

    .images {{ display: grid; grid-template-columns: repeat(2, minmax(160px, {thumb_size}px)); gap: 14px; margin-top: 8px; }}
    .images .card {{ background: #0e1319; border: 1px solid #1d2630; border-radius: 8px; padding: 6px; display: flex; align-items: center; justify-content: center; }}
    .images img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
    .placeholder {{ width: 100%; height: {thumb_size}px; display: grid; place-items: center; color: #708399; font-size: 12px; }}

    .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; margin-top: 10px; }}
    .gallery img {{ width: 100%; height: auto; border-radius: 6px; border: 1px solid #1d2630; }}
  </style>
</head>
<body>
  <header>
    <div class=\"title\">{html.escape(title)}</div>
    <div class=\"legend\">Rot = Overlap-Flag, Orange = Spread-Flag, Blau = Metadaten-Konflikt. Links: Oracle-Cluster-Representative. Rechts: nächster Zentroid-Nachbar.</div>
  </header>
  <main>
    {''.join(sections)}
  </main>
</body>
</html>
    """

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")
    return out_html

# ---------- Scatter plot ----------

def pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = Xc @ Vt[:2].T
    return Y.astype(np.float32)


def tsne_2d(X: np.ndarray, perplexity: float = 30.0, random_state: int = 0) -> np.ndarray:
    from sklearn.manifold import TSNE
    ts = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=random_state)
    Y = ts.fit_transform(X)
    return Y.astype(np.float32)


def render_scatter(
    out_html: Path,
    coords: np.ndarray,
    order: List[str],
    meta: Dict[str, Dict],
    flags: Dict[str, Dict],
    suspects_only: bool,
    thumb_dir: Path,
    thumb_size: int,
    title: str,
    cluster_report: Optional[str] = None,
) -> Path:
    X = coords.copy()
    if X.size == 0:
        X = np.zeros((0, 2), dtype=np.float32)
    mins = X.min(axis=0) if X.shape[0] else np.array([0.0, 0.0], dtype=np.float32)
    maxs = X.max(axis=0) if X.shape[0] else np.array([1.0, 1.0], dtype=np.float32)
    span = np.clip(maxs - mins, 1e-9, None)
    Xn = (X - mins) / span

    points: List[Dict] = []
    for idx, oid in enumerate(order):
        m = meta.get(oid, {})
        f = flags.get(oid, {})
        if suspects_only:
            if not any(int(f.get(k, 0)) for k in ("suspect_overlap","suspect_cluster_spread","suspect_metadata_conflict")):
                continue
        rep = m.get("image_path") or (m.get("image_paths_all") or [None])[0]
        thumb = _make_thumbnail(rep, thumb_dir, thumb_size) if rep else None
        points.append({
            "oid": oid,
            "name": m.get("name") or oid,
            "set_code": m.get("set_code") or "",
            "set_name": m.get("set_name") or "",
            "cmc": m.get("cmc"),
            "mana_cost": m.get("mana_cost") or "",
            "x": float(Xn[idx,0]),
            "y": float(Xn[idx,1]),
            "thumb": (os.path.relpath(thumb, out_html.parent) if thumb else None),
            "flags": {
                "overlap": int(f.get("suspect_overlap",0)),
                "spread": int(f.get("suspect_cluster_spread",0)),
                "meta": int(f.get("suspect_metadata_conflict",0)),
            }
        })

    data_json = json.dumps(points, ensure_ascii=False)
    cluster_rel = None
    if cluster_report:
        try:
            cluster_rel = os.path.relpath(cluster_report, out_html.parent)
        except Exception:
            cluster_rel = cluster_report

    doc = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{ --bg:#0b0d10; --fg:#d9e1ea; --muted:#9fb0c3; --panel:#121820; --line:#1d2630; }}
    body {{ margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background:var(--bg); color:var(--fg); }}
    header {{ position: sticky; top:0; z-index: 2; background:#10151b; border-bottom:1px solid var(--line); padding:10px 16px; }}
    header .title {{ font-weight:600; }}
    header .legend {{ font-size:12px; color:var(--muted); margin-top:4px; }}
    main {{ padding: 8px 12px 32px; }}

    .viewport {{ position: relative; height: 80vh; border:1px solid var(--line); border-radius:10px; background: #0f141a; overflow: hidden; }}
    svg {{ width: 100%; height: 100%; display: block; }}

    .point {{ cursor: pointer; opacity: 0.9; }}
    .point:hover {{ opacity: 1.0; }}
    .overlap {{ fill:#ff6262; }}
    .spread {{ fill:#ffa94d; }}
    .meta {{ fill:#70d5ff; }}
    .normal {{ fill:#9fb0c3; }}

    .tooltip {{ position: absolute; pointer-events: none; background: var(--panel); border:1px solid var(--line); border-radius:10px; padding:8px; display:none; width: 320px; box-shadow: 0 10px 30px rgba(0,0,0,.4); }}
    .tooltip .row {{ display:flex; gap:10px; }}
    .tooltip img {{ width: 96px; height:auto; border-radius:8px; border:1px solid var(--line); }}
    .tooltip .info {{ font-size:12px; color: var(--muted); }}
    .tooltip .name {{ color: var(--fg); font-weight:600; margin-bottom:2px; }}
  </style>
</head>
<body>
  <header>
    <div class=\"title\">{html.escape(title)}</div>
    <div class=\"legend\">Rot = Overlap, Orange = Spread, Blau = Metadaten-Konflikt. Hover: Bild/Details. Klick: öffnet Cluster-Report-Sektion (falls verlinkt).</div>
  </header>
  <main>
    <div class=\"viewport\">
      <svg viewBox=\"0 0 1000 600\" preserveAspectRatio=\"xMidYMid meet\" id=\"plot\"></svg>
      <div class=\"tooltip\" id=\"tooltip\"></div>
    </div>
  </main>
  <script>
    const data = {data_json};
    const svg = document.getElementById('plot');
    const tip = document.getElementById('tooltip');
    const W = 1000, H = 600;
    const clusterReport = {json.dumps(cluster_rel) if cluster_rel else 'null'};

    // Pan & Zoom state
    let scale = 1.0, tx = 0.0, ty = 0.0;
    let panning = false, lastX = 0, lastY = 0;

    function classForFlags(f) {{
      if (f.overlap) return 'point overlap';
      if (f.spread) return 'point spread';
      if (f.meta) return 'point meta';
      return 'point normal';
    }}

    function draw() {{
      svg.innerHTML = '';
      const root = document.createElementNS('http://www.w3.org/2000/svg','g');
      root.setAttribute('id','layer');
      svg.appendChild(root);
      for (const p of data) {{
        const cx = 40 + p.x * (W-80);
        const cy = 40 + (1.0 - p.y) * (H-80);
        const g = document.createElementNS('http://www.w3.org/2000/svg','g');
        const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
        c.setAttribute('cx', cx);
        c.setAttribute('cy', cy);
        c.setAttribute('r', 6);
        c.setAttribute('class', classForFlags(p.flags));
        g.appendChild(c);
        g.addEventListener('mousemove', (ev) => showTip(ev, p));
        g.addEventListener('mouseleave', hideTip);
        g.addEventListener('click', () => {{
          if (clusterReport) {{
            const url = clusterReport + '#cluster-' + encodeURIComponent(p.oid);
            window.open(url, '_blank');
          }}
        }});
        root.appendChild(g);
      }}
      applyTransform();
    }}

    function applyTransform() {{
      const root = document.getElementById('layer');
      if (root) {{
        root.setAttribute('transform', `translate(${{tx}},${{ty}}) scale(${{scale}})`);
      }}
    }}

    function showTip(ev, p) {{
      const bounds = svg.getBoundingClientRect();
      tip.style.display = 'block';
      tip.style.left = (ev.clientX - bounds.left + 12) + 'px';
      tip.style.top = (ev.clientY - bounds.top + 12) + 'px';
      const imgTag = p.thumb ? ('<img src="' + p.thumb + '" />') : '';
      const mana = (p.mana_cost || '');
      const cmc = (p.cmc !== undefined && p.cmc !== null) ? p.cmc : '';
      tip.innerHTML = '<div class="row">' + imgTag +
        '<div class="info">' +
        '<div class="name">' + p.name + '</div>' +
        '<div>ID: ' + p.oid + '</div>' +
        '<div>Set: ' + p.set_code + ' ' + p.set_name + '</div>' +
        '<div>Mana: ' + mana + ' | CMC: ' + cmc + '</div>' +
        '</div></div>';
    }}

    function hideTip() {{ tip.style.display = 'none'; }}

    svg.addEventListener('wheel', (ev) => {{
      ev.preventDefault();
      const rect = svg.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      const my = ev.clientY - rect.top;
      const factor = (ev.deltaY < 0) ? 1.1 : 0.9;
      const newScale = Math.min(10, Math.max(0.2, scale * factor));
      tx = mx - (mx - tx) * (newScale / scale);
      ty = my - (my - ty) * (newScale / scale);
      scale = newScale;
      applyTransform();
    }}, {{ passive: false }});

    svg.addEventListener('mousedown', (ev) => {{
      panning = true; lastX = ev.clientX; lastY = ev.clientY;
    }});
    window.addEventListener('mousemove', (ev) => {{
      if (!panning) return;
      const dx = ev.clientX - lastX; const dy = ev.clientY - lastY;
      tx += dx; ty += dy; lastX = ev.clientX; lastY = ev.clientY; applyTransform();
    }});
    window.addEventListener('mouseup', () => {{ panning = false; }});

    draw();
  </script>
</body>
</html>
    """

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")
    return out_html

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Oracle-Cluster-Visualisierung: Cluster-Report und/oder 2D-Scatter")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--sqlite", type=str, default=None)
    ap.add_argument("--mode", type=str, default="analysis", choices=["analysis","runtime"])
    ap.add_argument("--scenario", type=str, default=None, help="Szenario-Name (Default aus config.database.scenario)")
    ap.add_argument("--view", type=str, default="both", choices=["cluster","scatter","both"], help="Welche Ansicht rendern")

    # Cluster options
    ap.add_argument("--cluster-out", type=str, default=None, help="Pfad für Cluster-Report HTML")
    # Standard: nur Verdachtsfälle anzeigen; per --cluster-show-all überschreibbar
    ap.add_argument("--cluster-suspects-only", action="store_true", default=True, help="Nur Verdachtsfälle im Cluster-Report (Standard)")
    ap.add_argument("--cluster-show-all", action="store_true", help="Alle Cluster anzeigen (überschreibt den Standard-Filter)")
    ap.add_argument("--cluster-max-per", type=int, default=12)
    ap.add_argument("--cluster-thumb", type=int, default=192)

    # Scatter options
    ap.add_argument("--scatter-out", type=str, default=None, help="Pfad für Scatter HTML")
    ap.add_argument("--scatter-method", type=str, default="pca", choices=["pca","tsne"])
    ap.add_argument("--scatter-perplexity", type=float, default=30.0)
    # Standard: nur Verdachtsfälle anzeigen; per --scatter-show-all überschreibbar
    ap.add_argument("--scatter-suspects-only", action="store_true", default=True, help="Nur Verdachtsfälle im Scatter (Standard)")
    ap.add_argument("--scatter-show-all", action="store_true", help="Alle Punkte im Scatter anzeigen (überschreibt den Standard-Filter)")
    ap.add_argument("--scatter-thumb", type=int, default=160)

    args = ap.parse_args()

    cfg = load_config(args.config)
    sqlite_path = Path(args.sqlite or cfg.get("database", {}).get("sqlite_path") or "tcg_database/database/karten.db")
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite-DB nicht gefunden: {sqlite_path}")
    scenario = args.scenario or cfg.get("database", {}).get("scenario") or "default"

    ensure_oracle_quality_table(str(sqlite_path))

    emb_dim = int(
        cfg.get("encoder", {}).get("emb_dim")
        or cfg.get("vector", {}).get("dimension")
        or cfg.get("model", {}).get("embed_dim", 1024)
    )

    # Load metas and flags
    grouped, meta = load_embeddings_grouped_by_oracle(str(sqlite_path), args.mode, emb_dim, scenario=scenario)
    flags_map = _read_oracle_flags_map(sqlite_path, scenario)

    # Output defaults
    debug_dir = Path(cfg.get("paths", {}).get("debug_dir", "./debug"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cluster_out = Path(args.cluster_out or (debug_dir / f"oracle_cluster_visual_{args.mode}_{ts}.html"))
    scatter_out = Path(args.scatter_out or (debug_dir / f"oracle_cluster_scatter_{args.scatter_method}_{args.mode}_{ts}.html"))

    gen_cluster = args.view in ("cluster","both")
    gen_scatter = args.view in ("scatter","both")

    produced_cluster: Optional[Path] = None
    produced_scatter: Optional[Path] = None

    # Bestimme Filter-Defaults (Standard: nur Verdachtsfälle)
    cluster_suspects_only = True
    if getattr(args, "cluster_show_all", False):
        cluster_suspects_only = False
    elif getattr(args, "cluster_suspects_only", False):
        cluster_suspects_only = True

    scatter_suspects_only = True
    if getattr(args, "scatter_show_all", False):
        scatter_suspects_only = False
    elif getattr(args, "scatter_suspects_only", False):
        scatter_suspects_only = True

    # A) Cluster report
    if gen_cluster:
        rows = _read_oracle_quality(sqlite_path, scenario)
        produced_cluster = render_cluster_report(
            out_html=cluster_out,
            rows=rows,
            meta=meta,
            suspects_only=bool(cluster_suspects_only),
            max_images=max(0, int(args.cluster_max_per)),
            thumb_size=max(64, int(args.cluster_thumb)),
            title=f"Oracle Cluster Visual ({args.mode})",
        )
        print(f"[OK] Cluster-Report geschrieben: {produced_cluster}")

    # B) Scatter
    if gen_scatter:
        # compute centroids
        keys: List[str] = []
        cents: List[np.ndarray] = []
        for oid, arr in grouped.items():
            if arr.size == 0:
                continue
            arr = arr.astype(np.float32, copy=False)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.clip(norms, 1e-12, None)
            c = l2_normalize(np.mean(arr, axis=0))
            keys.append(oid)
            cents.append(c)
        X = np.stack(cents, axis=0) if cents else np.zeros((0, emb_dim), dtype=np.float32)
        if args.scatter_method == "pca":
            Y = pca_2d(X)
        else:
            Y = tsne_2d(X, perplexity=float(args.scatter_perplexity))
        produced_scatter = render_scatter(
            out_html=scatter_out,
            coords=Y,
            order=keys,
            meta=meta,
            flags=flags_map,
            suspects_only=bool(scatter_suspects_only),
            thumb_dir=(scatter_out.parent / "html_assets" / "thumbnails"),
            thumb_size=max(64, int(args.scatter_thumb)),
            title=f"Oracle Cluster Scatter ({args.scatter_method}, {args.mode})",
            cluster_report=str(produced_cluster) if produced_cluster else None,
        )
        print(f"[OK] Scatter-HTML geschrieben: {produced_scatter}")

    # Summary
    if produced_cluster:
        print(f"Cluster: {produced_cluster}")
    if produced_scatter:
        print(f"Scatter: {produced_scatter}")


if __name__ == "__main__":
    main()
