from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.inference import NexaInference

app = FastAPI(title="Nexa Backend API", version="1.0.0")

# ✅ CORS (works with Vercel + localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https:\/\/.*\.vercel\.app$|^https:\/\/.*$|^http:\/\/localhost(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL: Optional[NexaInference] = None
MODEL_ERROR: Optional[str] = None

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)


def init_model_if_needed():
    global MODEL, MODEL_ERROR
    if MODEL is not None or MODEL_ERROR is not None:
        return

    try:
        # weights file is in repo under models/fusion_best.pt
        weights_path = os.getenv("NEXA_WEIGHTS_PATH", "models/fusion_best.pt")
        MODEL = NexaInference(weights_path=weights_path)
        MODEL_ERROR = None
    except Exception as e:
        MODEL = None
        MODEL_ERROR = f"{type(e).__name__}: {str(e)}"


@app.get("/health")
def health():
    init_model_if_needed()
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_error": MODEL_ERROR,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    init_model_if_needed()

    if MODEL is None:
        raise HTTPException(status_code=500, detail=f"Model not available: {MODEL_ERROR}")

    # ✅ Save uploaded file to disk (so inference gets a real path)
    suffix = Path(file.filename).suffix or ".mp4"
    tmp_path = TMP_DIR / f"{uuid.uuid4().hex}{suffix}"

    try:
        data = await file.read()
        tmp_path.write_bytes(data)

        result = MODEL.predict(str(tmp_path))
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

    finally:
        # cleanup
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
