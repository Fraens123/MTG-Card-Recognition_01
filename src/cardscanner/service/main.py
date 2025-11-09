from __future__ import annotations
import io
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from psycopg_pool import ConnectionPool

from ..config import settings
from ..model import load_encoder
from ..transforms import get_preprocess
from ..db import query_topk
from .camera import Camera
from .visualizer import save_match_visual


app = FastAPI(title="MTG Card Matcher")


model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocess = get_preprocess(224)
pool: Optional[ConnectionPool] = None
camera: Optional[Camera] = None


@app.on_event("startup")
def on_startup():
    global model, pool, camera
    model = load_encoder(settings.model_weights, embed_dim=settings.vector_dim, device=device)
    pool = ConnectionPool(conninfo=settings.database_url, min_size=1, max_size=4)
    try:
        camera = Camera()
    except Exception:
        camera = None


@app.on_event("shutdown")
def on_shutdown():
    global pool, camera
    if pool is not None:
        pool.close()
    if camera is not None:
        camera.release()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/match_card")
async def match_card(
    file: Optional[UploadFile] = File(default=None),
    use_camera: bool = Query(default=False),
    top_k: int = Query(default=settings.top_k_default, ge=1, le=50),
    threshold: Optional[float] = Query(default=None),
):
    if use_camera:
        if camera is None:
            return JSONResponse(status_code=400, content={"error": "Camera not available"})
        pil_img = camera.capture_image()
    else:
        if file is None:
            return JSONResponse(status_code=400, content={"error": "Provide file or set use_camera=true"})
        data = await file.read()
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")

    x = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_normalized(x).cpu().numpy().flatten().astype("float32")

    with pool.connection() as conn:
        with conn.cursor() as cur:
            matches = query_topk(cur, emb, top_k=top_k, threshold=threshold)

    vis_path = None
    if matches:
        vis_path = save_match_visual(pil_img, matches[0].get("image_path"))

    return {"matches": matches, "visualization_path": vis_path}

