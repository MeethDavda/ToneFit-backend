# app/services/segmentation.py
from __future__ import annotations
from typing import NamedTuple, Tuple

from typing import Optional, Tuple
# from app.services.pipeline import ColorFeat, _clip01, circular_diff_deg
import numpy as np
import cv2
import colorsys

# --- Hard dependency: rembg (+ onnxruntime or onnxruntime-silicon) ---
try:
    from rembg import remove as rembg_remove, new_session
    SESSION = new_session("u2netp") 
except Exception as e:
    raise ImportError(
        "rembg is required for background removal.\n"
        "Install:\n"
        "  pip install rembg onnxruntime\n"
        "Apple Silicon:\n"
        "  pip install rembg onnxruntime-silicon\n"
        f"Original import error: {e}"
    )

Mask = np.ndarray                  # uint8 [H,W], values {0,255}
BBox = Tuple[int, int, int, int]   # (x1, y1, x2, y2)

def _resize_max_side(img, max_side=1024):
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side: return img
    scale = max_side / side
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def remove_background(bgr: np.ndarray, alpha_threshold: int = 8) -> Mask:
    if bgr is None or bgr.size == 0:
        raise ValueError("remove_background: empty image")

    # ↓↓↓ shrink BEFORE rembg to cut memory/CPU
    bgr_small = _resize_max_side(bgr, max_side=1024)

    ok, buf = cv2.imencode(".png", bgr_small)
    if not ok:
        raise ValueError("PNG encode failed for rembg input.")

    # reuse the same session (don’t reload the model per request!)
    cutout_png = rembg_remove(buf.tobytes(), session=SESSION)
    rgba = _decode_rgba(cutout_png)
    if rgba is None or rgba.shape[2] < 4:
        raise RuntimeError("rembg did not return an RGBA image.")

    alpha = rgba[:, :, 3]
    mask = (alpha > alpha_threshold).astype(np.uint8) * 255
    return _clean_mask(mask)


def person_bbox(mask: Mask, min_area_frac: float = 0.02) -> Optional[BBox]:
    """
    Tight bbox around the largest connected component of mask.
    Returns None if no plausible person is found.
    """
    if mask is None or mask.size == 0:
        return None

    m = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    x, y, w, h, area = stats[idx, :5]

    H, W = mask.shape[:2]
    if area < (H * W * min_area_frac):
        return None

    pad_x = max(1, int(0.01 * W))
    pad_y = max(1, int(0.01 * H))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W - 1, x + w + pad_x - 1)
    y2 = min(H - 1, y + h + pad_y - 1)
    return (x1, y1, x2, y2)


def split_upper_lower(mask: Mask, bbox: BBox, split_ratio: float = 0.55) -> Tuple[Mask, Mask]:
    """
    Split the bbox horizontally at split_ratio (0..1 from top) and intersect with mask.
    Returns (upper_mask, lower_mask) as uint8 {0,255}.
    """
    x1, y1, x2, y2 = bbox
    H, W = mask.shape[:2]

    # clamp bbox
    x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    height = max(1, y2 - y1 + 1)
    split_y = int(y1 + split_ratio * height)
    split_y = max(y1 + 1, min(split_y, y2 - 1))  # ensure both halves non-empty

    upper = np.zeros_like(mask, dtype=np.uint8)
    lower = np.zeros_like(mask, dtype=np.uint8)
    upper[y1:split_y, x1:x2 + 1] = 255
    lower[split_y:y2 + 1, x1:x2 + 1] = 255

    upper = cv2.bitwise_and(upper, mask)
    lower = cv2.bitwise_and(lower, mask)

    return _clean_mask(upper), _clean_mask(lower)



class ColorFeat(NamedTuple):
    rgb: Tuple[int, int, int]     # (R,G,B) 0..255
    hue_deg: float                # 0..360
    sat: float                    # 0..1
    lightness_L: float            # ~0..100 (OpenCV Lab L*)

def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def circular_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d

# ---------- helpers ----------

def _decode_rgba(png_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

def _clean_mask(mask: Mask) -> Mask:
    """Morph open/close + keep largest component to reduce halos/holes."""
    if mask is None or mask.size == 0:
        return mask
    m = (mask > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if num <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    keep = (labels == idx).astype(np.uint8) * 255
    return keep

def _hue_name(h: float) -> str:
    """Rough human-friendly names for hue degrees."""
    h = h % 360
    bins = [
        (345, 360, "red"), (0, 15, "red"),
        (15, 45, "orange/amber"),
        (45, 65, "yellow"),
        (65, 85, "lime/chartreuse"),
        (85, 150, "green"),
        (150, 190, "teal/cyan"),
        (190, 230, "blue"),
        (230, 270, "indigo/violet"),
        (270, 300, "magenta"),
        (300, 345, "pink/fuchsia"),
    ]
    for lo, hi, name in bins:
        if lo <= hi and lo <= h < hi:
            return name
        if lo > hi and (h >= lo or h < hi):  # wraparound for red sector
            return name
    return "color"

def _hex_from_hsv(h_deg: float, s: float, v: float) -> str:
    """h in degrees, s/v in [0,1]."""
    r, g, b = colorsys.hsv_to_rgb((h_deg % 360) / 360.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return f"#{int(round(r*255)):02X}{int(round(g*255)):02X}{int(round(b*255)):02X}"

def _nice_accent_from(h_deg: float, s: float, L: float, mode: str = "complementary") -> tuple[str, str]:
    """
    Suggest an accent hue/hex from a base hue. `mode` in {"complementary", "triadic", "analogous"}.
    Returns (name, hex).
    """
    if mode == "triadic":
        # pick the farther of the two triadic points to maximize separation from bottom
        accent_h = (h_deg + 120.0) % 360
    elif mode == "analogous":
        accent_h = (h_deg + 30.0) % 360
    else:  # complementary
        accent_h = (h_deg + 180.0) % 360

    # Choose a reasonable S/V based on current piece
    # If lightness is high, slightly lower V so the accent doesn’t blow out; otherwise brighten it
    v = 0.62 if L >= 60 else 0.72
    s = max(0.55, min(0.9, s + 0.15))
    return _hue_name(accent_h), _hex_from_hsv(accent_h, s, v)

def _format_item_list(items: list[str]) -> str:
    return ", ".join(items[:-1]) + f", or {items[-1]}" if len(items) > 1 else items[0]

def _is_neutral(s: float, thresh: float = 0.20) -> bool:
    return s < thresh

def _is_warm_neutral(h: float, s: float) -> bool:
    # beige/cream/tan family
    return _is_neutral(s) and 20 <= (h % 360) <= 80

def _is_cool_neutral(h: float, s: float) -> bool:
    # slate/steel/charcoal/denim-ish neutrals
    h = h % 360
    return _is_neutral(s) and (180 <= h <= 260 or s < 0.08)

def score_tonal_neutral(top: ColorFeat, bottom: ColorFeat) -> float:
    """
    Reward subtle, low-saturation pairings (e.g., cream + khaki, grey + black)
    with *moderate* lightness spread and small hue separation.
    Peak around ΔL* ≈ 25 and ΔHue ≤ 30°.
    """
    s1, s2 = top.sat, bottom.sat
    if not (_is_neutral(s1) and _is_neutral(s2)):
        return 0.0

    d_h = circular_diff_deg(top.hue_deg, bottom.hue_deg)
    # hue term: best when close (≤30°)
    hue_term = max(0.0, 1.0 - min(d_h, 30.0) / 30.0)

    # lightness term: bell-shaped around 25 (good separation without harsh contrast)
    dL = abs(top.lightness_L - bottom.lightness_L)
    light_term = max(0.0, 1.0 - abs(dL - 25.0) / 25.0)  # 1 at 25, ~0 at 0 or 50

    # warmth/coolness cohesion bonus (e.g., cream+khaki or grey+black)
    cohesive = 1.0 if (_is_warm_neutral(top.hue_deg, s1) and _is_warm_neutral(bottom.hue_deg, s2)) \
                    or (_is_cool_neutral(top.hue_deg, s1) and _is_cool_neutral(bottom.hue_deg, s2)) else 0.7

    return _clip01(0.6 * hue_term + 0.4 * light_term) * cohesive
