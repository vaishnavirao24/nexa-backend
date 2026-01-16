import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ===== IMPORT YOUR INFERENCE CLASS =====
from src.inference import NexaInference

# ===== FASTAPI APP =====
app = FastAPI(title="Nexa Backend API")

# ===== CORS (VERY IMPORTANT FOR FRONTEND) =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://hello-nexa.lovable.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MODEL PATHS (ABSOLUTE, SAFE) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FUSION_MODEL_PATH = os.path.join(BASE_DIR, "models", "fusion_best.pt")
YOLO_POSE_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-pose.pt")

# ===== LOAD MODEL SAFELY (NO CRASH) =====
try:
    nexa = NexaInference(
        weights_path=FUSION_MODEL_PATH,
        yolo_pose_path=YOLO_POSE_MODEL_PATH,
        device="cpu"  # Render is CPU only
    )
    MODEL_LOADED = True
    print("✅ Model loaded successfully")
except Exception as e:
    nexa = None
    MODEL_LOADED = False
    print("❌ Model failed to load:", e)

# ===== HEALTH CHECK =====
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED
    }

# ===== PREDICTION ENDPOINT =====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL_LOADED:
        return JSONResponse(
            status_code=500,
            content={"error": "Model not loaded"}
        )

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = nexa.predict(tmp_path)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
