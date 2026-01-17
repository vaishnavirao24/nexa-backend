# app.py
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from src.inference import NexaInference

app = FastAPI(title="Nexa Backend API", version="1.0.0")

# -----------------------------
# CORS (IMPORTANT for Vercel)
# -----------------------------
# Add your exact Vercel domains here so the browser will allow responses.
# (You can keep localhost for local testing.)
ALLOWED_ORIGINS = [
    "https://hello-nexa.vercel.app",
    # If you use Vercel preview deployments, keep these patterns by adding each preview URL you see.
    # Example from your screenshot (keep it if it exists for your project):
    "https://hello-nexa-git-main-vaishnavis-projects-1e89fb17.vercel.app",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL: Optional[NexaInference] = None
MODEL_ERROR: Optional[str] = None


def init_model_if_needed() -> None:
    """
    Lazily initialize the model once.
    If it fails, we keep the server alive and expose error in /health.
    """
    global MODEL, MODEL_ERROR

    if MODEL is not None or MODEL_ERROR is not None:
        return

    try:
        # If you want to load weights in Render, set this env var in Render dashboard:
        # NEXA_WEIGHTS_PATH=/opt/render/project/src/models/fusion_best.pt   (example)
        weights_path = os.getenv("NEXA_WEIGHTS_PATH")

        # Optional: you might have a YOLO pose path env var too
        yolo_pose_path = os.getenv("NEXA_YOLO_POSE_PATH")

        # Optional: device (Render free is CPU)
        device = os.getenv("NEXA_DEVICE", "cpu")

        # IMPORTANT:
        # Your NexaInference __init__ must match these args.
        # If your class supports only weights_path, it will ignore the others if you design it that way.
        MODEL = NexaInference(
            weights_path=weights_path,
            yolo_pose_path=yolo_pose_path,
            device=device,
        )
        MODEL_ERROR = None

    except Exception as e:
        MODEL = None
        MODEL_ERROR = f"{type(e).__name__}: {str(e)}"


@app.get("/health")
def health():
    """
    Used by Render + quick debug.
    """
    # ensure init has been attempted at least once if you want
    # init_model_if_needed()

    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_error": MODEL_ERROR,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts multipart/form-data with a single file field called 'file'.
    Frontend should POST to /predict and attach the video as `file`.
    """
    init_model_if_needed()

    video_bytes = await file.read()

    if MODEL is None:
        # Fallback response (keeps frontend alive even if model isn't loaded)
        return {
            "predicted_class": "pose_idle",
            "confidence": 0.0,
            "class_probabilities": {
                "emotion": 0.0,
                "social": 0.0,
                "physical": 0.0,
                "pose_idle": 1.0,
            },
            "note": "Model not available on server",
            "model_error": MODEL_ERROR,
        }

    # Your inference.py must implement: predict(video_bytes: bytes) -> dict
    return MODEL.predict(video_bytes)
