"""Optional photo clean-up that runs before detection (OpenCV).
It only acts when a photo is dark, flat or soft. It cannot recreate detail that
the camera never captured, so very blurry photos will still be hard."""
import cv2
import numpy as np
from PIL import Image


def enhance_if_needed(img):
    bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = 1000 / max(gray.shape)
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else gray
    mean, std = float(small.mean()), float(small.std())
    sharp = float(cv2.Laplacian(small, cv2.CV_32F).var())
    steps = []
    if mean < 90:
        gamma = 0.6 if mean < 60 else 0.8
        lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype("uint8")
        bgr = cv2.LUT(bgr, lut)
        steps.append("brightened")
    if std < 45 or mean < 90:
        l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        steps.append("contrast boosted")
    if sharp < 40:
        soft = cv2.GaussianBlur(bgr, (0, 0), 2.0)
        bgr = cv2.addWeighted(bgr, 1.8, soft, -0.8, 0)
        steps.append("sharpened")
    if not steps:
        return img, []
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), steps