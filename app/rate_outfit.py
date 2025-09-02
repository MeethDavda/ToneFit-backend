from typing import List, Optional, Tuple
import io
import os
import cv2
import math

import numpy as np
from  fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from rembg import remove as rembg_remove
from app.services.pipeline import rate_outfit

router = APIRouter(tags=["outfit"])

IMG_PATH = os.path.join(os.path.dirname(__file__),"abhi.png")
IMG_OUT = os.path.join(os.path.dirname(__file__),"out.png")

@router.post("/rate-outfits")
async def rate_outfits(image:UploadFile = File(...)):
    img_bytes = await image.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # with open(IMG_PATH, 'rb') as i:
    #     with open(IMG_OUT, 'wb') as o:
    #         input = i.read()
    #         output = rembg_remove(input)
    #         o.write(output)

    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not read the image.")

    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image.")
    try:
        test = rate_outfit(buf.tobytes())
        print(test)
        return test
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



