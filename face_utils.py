import os
import cv2
import dlib
from variables import *

_predictor = None
if os.path.exists(LANDMARK_MODEL):
    _predictor = dlib.shape_predictor(LANDMARK_MODEL)
else:
    print(f"[Warning] '{LANDMARK_MODEL}' not found - using plain crops without alignment.")


def scale_rect(rect, ratio):
    """Map a rectangle found on a downscaled frame back to full-resolution coordinates."""
    return dlib.rectangle(
        int(rect.left() / ratio), int(rect.top() / ratio),
        int(rect.right() / ratio), int(rect.bottom() / ratio),
    )


def largest_face(faces):
    return max(faces, key=lambda f: f.width() * f.height())


def extract_face(frame_bgr, rect):
    """Return a BGR face image ready for get_face_embedding, or None.

    Aligned (eye-levelled, scaled) when the landmark model is available,
    otherwise a clamped crop of the detector box.
    """
    h, w = frame_bgr.shape[:2]

    if _predictor is not None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        shape = _predictor(rgb, rect)
        chip = dlib.get_face_chip(rgb, shape, size=FACE_SIZE, padding=FACE_PADDING)
        return cv2.cvtColor(chip, cv2.COLOR_RGB2BGR)

    x1, y1 = max(0, rect.left()), max(0, rect.top())
    x2, y2 = min(w, rect.right()), min(h, rect.bottom())
    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size > 0 else None
