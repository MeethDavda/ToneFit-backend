# app/services/pipeline.py
from __future__ import annotations

from typing import Optional, Tuple, List
from collections import namedtuple

import numpy as np
import cv2
from sklearn.cluster import KMeans  # pip install scikit-learn

from app.services.segmentation import remove_background, person_bbox, split_upper_lower,_nice_accent_from,_format_item_list,_hue_name,score_tonal_neutral


# ========= Public API =========

def rate_outfit(img_bytes: bytes) -> dict:
    """
    Orchestrates the rule-based outfit color check:
      bytes -> decode -> background removal -> bbox -> upper/lower masks
      -> dominant colors -> rule scores -> verdict + tips

    Returns a JSON-serializable dict.
    """
    bgr = _decode_image(img_bytes)
    if bgr is None:
        raise ValueError("Could not decode image data.")

    # Resize very large images for speed/memory
    bgr = _resize_max_side(bgr, max_side=1280)

    # Segmentation
    mask = remove_background(bgr)
    bbox = person_bbox(mask)
    if bbox is None:
        raise ValueError("No person detected after background removal.")

    upper_mask, lower_mask = split_upper_lower(mask, bbox, split_ratio=0.55)

    # Dominant color features for top/bottom
    top = dominant_color_features(bgr, upper_mask, k=3)
    bottom = dominant_color_features(bgr, lower_mask, k=3)
    if top is None or bottom is None:
        raise ValueError("Could not determine dominant colors for clothing regions.")

    # Score + verdict + tips
    score, breakdown = blended_score(top, bottom)
    verdict = label(score)
    tips = build_tips(top, bottom, breakdown)

    return {
        "verdict": verdict,
        "score": round(float(score), 3),
        "tips": tips,
        "top_hex": rgb_to_hex(top.rgb),
        "bottom_hex": rgb_to_hex(bottom.rgb),
        "debug": {
            "top": top._asdict(),
            "bottom": bottom._asdict(),
            "breakdown": breakdown,
        },
    }


# ========= Image utils =========

def _decode_image(img_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _resize_max_side(img: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img
    scale = max_side / side
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ========= Color extraction =========

ColorFeat = namedtuple("ColorFeat", ["rgb", "hue_deg", "sat", "lightness_L"])

def dominant_color_features(bgr_img: np.ndarray, mask: np.ndarray, k: int = 3) -> Optional[ColorFeat]:
    """
    Returns dominant color features (RGB, hue°, saturation [0..1], Lab L* [0..100])
    for pixels inside `mask`.
    """
    px = _mask_pixels(bgr_img, mask)
    if px.size == 0:
        return None

    dom_bgr = _kmeans_dom_bgr(px, k=k)
    if dom_bgr is None:
        return None

    dom_bgr = np.clip(dom_bgr, 0, 255).astype(np.uint8)
    dom_rgb = dom_bgr[::-1]  # BGR -> RGB
    rgb_tuple = (int(dom_rgb[0]), int(dom_rgb[1]), int(dom_rgb[2]))

    hsv = cv2.cvtColor(dom_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV).reshape(3,)
    h_deg = float(hsv[0]) * 2.0          # OpenCV hue: 0..179 -> 0..358 °
    s = float(hsv[1]) / 255.0            # 0..1

    lab = cv2.cvtColor(dom_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).reshape(3,)
    L = float(lab[0])                    # OpenCV L* approx 0..100
    return ColorFeat(rgb=rgb_tuple, hue_deg=h_deg, sat=s, lightness_L=L)

def _mask_pixels(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    return img[ys, xs, :]


def _kmeans_dom_bgr(pixels_bgr: np.ndarray, k: int = 3) -> Optional[np.ndarray]:
    """
    K-means in 8-bit Lab (L,a,b in 0..255). Convert centers back to BGR using uint8.
    """
    if pixels_bgr.size == 0:
        return None

    # Subsample for speed on big regions
    if len(pixels_bgr) > 40000:
        idx = np.random.choice(len(pixels_bgr), 40000, replace=False)
        pixels_bgr = pixels_bgr[idx]

    # Convert to 8-bit Lab (OpenCV's uint8 Lab uses L,a,b in 0..255)
    lab_u8 = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)

    # KMeans expects float; that's fine. Centers will be in ~0..255 too.
    km = KMeans(n_clusters=min(k, len(lab_u8)), n_init="auto", random_state=42)
    labels = km.fit_predict(lab_u8.astype(np.float32))
    centers_lab = km.cluster_centers_  # float32 ~0..255

    # Dominant cluster
    counts = np.bincount(labels)
    dom_idx = int(np.argmax(counts))
    dom_lab_u8 = np.clip(centers_lab[dom_idx], 0, 255).astype(np.uint8)

    # Convert back to BGR, but keep dtype consistent with 8-bit Lab
    dom_bgr = cv2.cvtColor(dom_lab_u8.reshape(1, 1, 3), cv2.COLOR_LAB2BGR).reshape(3,)
    return dom_bgr


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [int(max(0, min(255, v))) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


# ========= Scoring (rules) =========

def circular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference on a 0..360° hue circle."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)

def score_analogous(h1: float, h2: float, tol: float = 30.0) -> float:
    d = circular_diff_deg(h1, h2)
    return _clip01(1.0 - min(d, tol) / tol)

def score_complementary(h1: float, h2: float, tol: float = 20.0) -> float:
    d = abs(circular_diff_deg(h1, (h2 + 180.0) % 360.0))
    return _clip01(1.0 - min(d, tol) / tol)

def score_triadic(h1: float, h2: float, tol: float = 20.0) -> float:
    d1 = abs(circular_diff_deg(h1, (h2 + 120.0) % 360.0))
    d2 = abs(circular_diff_deg(h1, (h2 + 240.0) % 360.0))
    d = min(d1, d2)
    return _clip01(1.0 - min(d, tol) / tol)

def score_monochrome(h1: float, h2: float, tol: float = 12.0) -> float:
    d = circular_diff_deg(h1, h2)
    return _clip01(1.0 - min(d, tol) / tol)

def contrast_bonus(L1: float, L2: float, deltaL_bonus: float = 25.0) -> float:
    return 0.2 if abs(L1 - L2) >= deltaL_bonus else 0.0

def neutral_bonus(s1: float, s2: float, L1: float, L2: float, neutral_s: float = 0.15) -> float:
    """
    If any item is near-neutral (low saturation), be lenient.
    Extra small bonus if lightness contrast is okay.
    """
    if s1 < neutral_s or s2 < neutral_s:
        return 0.15 + (0.1 if abs(L1 - L2) >= 20 else 0.0)
    return 0.0

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def blended_score(top: ColorFeat, bottom: ColorFeat) -> Tuple[float, dict]:
    h1, h2 = top.hue_deg, bottom.hue_deg
    L1, L2 = top.lightness_L, bottom.lightness_L
    s1, s2 = top.sat, bottom.sat

    s_analog = score_analogous(h1, h2)
    s_comp   = score_complementary(h1, h2)
    s_triad  = score_triadic(h1, h2)
    s_mono   = score_monochrome(h1, h2)
    s_tonal  = score_tonal_neutral(top, bottom)  # NEW

    # Tonal-neutral can now win
    base = max(s_analog, s_comp, s_triad, s_mono, s_tonal)

    # Bonuses (avoid stacking too much on neutral looks)
    bonus = contrast_bonus(L1, L2) + neutral_bonus(s1, s2, L1, L2)
    if s_tonal >= 0.6:
        bonus = max(0.0, bonus - 0.1)  # small de-emphasis to keep scores realistic

    final = _clip01(base + bonus)

    breakdown = {
        "analogous": round(s_analog, 3),
        "complementary": round(s_comp, 3),
        "triadic": round(s_triad, 3),
        "monochrome": round(s_mono, 3),
        "tonal_neutral": round(s_tonal, 3),     # NEW
        "contrast_bonus": round(contrast_bonus(L1, L2), 3),
        "neutral_bonus": round(neutral_bonus(s1, s2, L1, L2), 3),
        "base": round(base, 3),
        "final": round(final, 3),
    }
    return final, breakdown


def label(score: float) -> str:
    if score >= 0.65:
        return "GOOD"
    if score >= 0.45:
        return "BORDERLINE"
    return "NOT_GREAT"


# ========= Tips =========

def build_tips(top: ColorFeat, bottom: ColorFeat, breakdown: dict) -> List[str]:
    tips: List[str] = []
    d_hue = circular_diff_deg(top.hue_deg, bottom.hue_deg)
    d_L   = abs(top.lightness_L - bottom.lightness_L)
    both_neutral = (top.sat < 0.20 and bottom.sat < 0.20)

    if breakdown.get("tonal_neutral", 0.0) >= 0.6 and both_neutral:
        tips.append(
            "Clean tonal palette — keep it subtle. Use material contrast (fleece/knit vs. twill/denim) "
            "and a warm metal (gold) or leather accent for depth."
        )
        # If lightness spread is small, offer a gentle tweak
        if d_L < 18:
            tips.append("Slightly increase separation (ΔL* ≈ 20–30): a lighter tee/hoodie or slightly darker chinos.")
        # Echo small black elements if present (sunglasses/shoe logo)
        tips.append("Echo small dark accents (sunglasses, watch strap, shoe logo) to tie the look together.")
        return tips[:3]

    # Low base + low contrast → give a concrete fix
    if breakdown["base"] < 0.5 and breakdown["contrast_bonus"] < 0.2:
        direction = "lighter top" if top.lightness_L < bottom.lightness_L else "darker bottom"
        tips.append(
            f"Low contrast is flattening the look—raise separation by ~ΔL* 25: try a {direction} or swap one piece to a brighter neutral."
        )

    # If analogous is strong, suggest texture or a tiny third color (not loud)
    if breakdown["analogous"] >= 0.6 and d_hue <= 35:
        tips.append(
            "Analogous harmony—add texture (rib knit, suede, denim) or a tiny third neutral (cream/stone/charcoal) for dimension."
        )

    # Complementary/triadic are weak → offer a muted accent (only if not both neutral)
    if not both_neutral and breakdown["complementary"] < 0.5 and breakdown["triadic"] < 0.5:
        tips.append(
            "Add a small muted accent (belt or hat) rather than a bright pop to keep the palette refined."
        )

    # Both highly saturated? Suggest softening one
    if top.sat >= 0.6 and bottom.sat >= 0.6:
        tips.append("Both garments are very saturated—soften one (washed tone or muted neutral) to reduce visual competition.")

    # Monochrome with low ΔL* → suggest texture/spread
    if breakdown["monochrome"] >= 0.7 and d_L < 20:
        tips.append("Monochrome pairing—add light/dark spread (ΔL* ≈ 25) or mix textures (knit vs. denim vs. leather).")

    if not tips:
        # fallback positive reinforcement
        major = max(
            [("analogous", breakdown["analogous"]),
             ("tonal neutral", breakdown.get("tonal_neutral", 0.0)),
             ("complementary", breakdown["complementary"]),
             ("triadic", breakdown["triadic"]),
             ("monochrome", breakdown["monochrome"])],
            key=lambda x: x[1],
        )[0]
        tips.append(f"Nice {major} harmony—lock it in with a subtle material contrast and minimal accessories.")
    return tips[:3]

