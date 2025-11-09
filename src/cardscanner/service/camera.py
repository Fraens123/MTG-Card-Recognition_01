from __future__ import annotations
from typing import Optional
import numpy as np
from PIL import Image


class Camera:
    def __init__(self):
        self._mode = None
        self._cap = None
        try:
            from picamera2 import Picamera2  # type: ignore
            self._mode = "picamera2"
            self._cap = Picamera2()
            self._cap.configure(self._cap.create_preview_configuration())
            self._cap.start()
        except Exception:
            import cv2  # type: ignore
            self._mode = "opencv"
            self._cap = cv2.VideoCapture(0)

    def capture_image(self) -> Image.Image:
        if self._mode == "picamera2":
            arr = self._cap.capture_array()
            img = Image.fromarray(arr[..., ::-1])
            return img.convert("RGB")
        else:
            import cv2  # type: ignore
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError("Failed to capture from camera")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)

    def release(self):
        try:
            if self._mode == "opencv" and self._cap is not None:
                self._cap.release()
            if self._mode == "picamera2" and self._cap is not None:
                self._cap.stop()
        except Exception:
            pass
