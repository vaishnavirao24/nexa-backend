from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import uuid

from src.inference import NexaInference

# -----------------------------
# App init
# -----------------------------
app = FastAPI(title="Nexa API", version="1.0")

# -----------------------------
# CORS (VERY IMPORTANT)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hello-nexa.lovable.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Model init (load once)
# -----------------------------
nexa = NexaInference()

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"status": "Nexa backend running"}

# -----------------------------
# Inference endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Create temp directory if not exists
        os.makedirs("tmp", exist_ok=True)

        # Save uploaded file
        file_ext = os.path.splitext(file.filename)[1]
        temp_path = f"tmp/{uuid.uuid4()}{file_ext}"

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run inference
        result = nexa.predict(temp_path)

        # Cleanup
        os.remove(temp_path)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
