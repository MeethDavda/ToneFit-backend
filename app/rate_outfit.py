# app/rate_outfit.py
from fastapi import APIRouter, File, HTTPException, UploadFile
import asyncio
import numpy as np
import cv2

from app.services.pipeline import rate_outfit

router = APIRouter(tags=["outfit"])

# --- keep OpenCV lean ---
cv2.setNumThreads(1)

# --- simple in-process queue so only 1 heavy job runs at a time ---
HEAVY_OP = asyncio.Semaphore(1)
QUEUE_TIMEOUT_SEC = 25

MAX_MB = 8  # reject very large uploads


def _downscale_guard(bgr, max_side=896):
    h, w = bgr.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return bgr
    s = max_side / side
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


@router.post("/rate-outfits")
async def rate_outfits(image: UploadFile = File(...)):
    # 1) quick validations
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Use JPEG or PNG.")
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(img_bytes) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image too large (>{MAX_MB} MB).")

    # 2) decode -> downscale (hard cap) to avoid OOM on huge phone photos
    np_arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image bytes.")
    bgr = _downscale_guard(bgr, max_side=1024)

    # 3) queue heavy work so only one request runs the model at a time
    try:
        await asyncio.wait_for(HEAVY_OP.acquire(), timeout=QUEUE_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server busy, try again in a moment.")
    try:
        # 4) encode -> pipeline (pipeline expects encoded image bytes)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode image.")
        return rate_outfit(buf.tobytes())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    finally:
        HEAVY_OP.release()
