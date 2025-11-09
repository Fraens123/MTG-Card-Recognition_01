import os
from datetime import datetime
from typing import Optional
from PIL import Image

from ..config import settings


def save_match_visual(camera_image: Image.Image, scryfall_path: Optional[str], prefix: str = "match") -> str:
    os.makedirs(settings.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cam_out = os.path.join(settings.output_dir, f"{prefix}_{ts}_cam.jpg")
    camera_image.save(cam_out, quality=92)

    if scryfall_path and os.path.exists(scryfall_path):
        try:
            scr_img = Image.open(scryfall_path).convert("RGB")
            w = max(camera_image.width, scr_img.width)
            h = max(camera_image.height, scr_img.height)
            cam_resized = camera_image.resize((w, h))
            scr_resized = scr_img.resize((w, h))
            side_by_side = Image.new("RGB", (w * 2, h))
            side_by_side.paste(cam_resized, (0, 0))
            side_by_side.paste(scr_resized, (w, 0))
            sbs_out = os.path.join(settings.output_dir, f"{prefix}_{ts}_compare.jpg")
            side_by_side.save(sbs_out, quality=92)
            return sbs_out
        except Exception:
            return cam_out
    return cam_out
