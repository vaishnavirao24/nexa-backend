from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.inference import NexaInference


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "fusion_best.pt"
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

# ------------------------
# App
# ------------------------
app = FastAPI(title="Nexa API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: NexaInference | None = None


# ------------------------
# Startup
# ------------------------
@app.on_event("startup")
def startup():
    global engine
    engine = NexaInference(weights_path=MODEL_PATH)


# ------------------------
# Routes
# ------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global engine
    if engine is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".mkv", ".avi"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    ts = int(time.time() * 1000)
    temp_path = TMP_DIR / f"upload_{ts}{ext}"
    temp_path.write_bytes(content)

    try:
        t0 = time.time()
        res = engine.predict(str(temp_path), top_k=3)
        total_ms = int((time.time() - t0) * 1000)

        return {
            "ok": True,
            "predicted_label": res.predicted_label,
            "confidence": res.confidence,
            "topk": res.topk,
            "timings_ms": {"total": total_ms},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    finally:
        temp_path.unlink(missing_ok=True)
