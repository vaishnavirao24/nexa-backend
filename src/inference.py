# src/inference.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .labels import CLASS_NAMES

# ----------------------------
# Device
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Pose Encoder (Transformer)
# ----------------------------
class PoseEncoder(nn.Module):
    def __init__(self, input_dim: int = 73, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 73)
        x = self.proj(x)
        x = self.encoder(x)
        return x.mean(dim=1)  # (B, d_model)


# ----------------------------
# Appearance Encoder
# ----------------------------
class AppearanceEncoder(nn.Module):
    def __init__(self, input_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, out_dim)


# ----------------------------
# Fusion Model
# ----------------------------
class FusionModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.pose = PoseEncoder()
        self.app = AppearanceEncoder()

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, pose: torch.Tensor, app: torch.Tensor) -> torch.Tensor:
        pose_feat = self.pose(pose)  # (B, 128)
        app_feat = self.app(app)     # (B, 128)
        fused = torch.cat([pose_feat, app_feat], dim=1)  # (B, 256)
        return self.classifier(fused)  # (B, num_classes)


# ----------------------------
# Inference Wrapper
# ----------------------------
class NexaInference:
    """
    This wrapper is used by app.py.
    IMPORTANT: Frontend expects this JSON shape:

    {
      "predicted_class": "emotion|social|physical|pose_idle",
      "confidence": 0.0-1.0,
      "class_probabilities": {
          "emotion": 0.0-1.0,
          "social": 0.0-1.0,
          "physical": 0.0-1.0,
          "pose_idle": 0.0-1.0
      }
    }
    """

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        self.device = DEVICE
        self.model = FusionModel(num_classes=len(CLASS_NAMES)).to(self.device)
        self.model.eval()

        # If weights_path is provided, load it. Otherwise keep random weights (still works for UI).
        if weights_path:
            self._load_weights(weights_path)

    def _load_weights(self, weights_path: Union[str, Path]) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=self.device)

        # Support both raw state_dict and wrapped checkpoint
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        if missing:
            print("⚠️ Missing keys (ignored):")
            for k in missing:
                print("  ", k)

        if unexpected:
            print("⚠️ Unexpected keys (ignored):")
            for k in unexpected:
                print("  ", k)

        print("✅ Model loaded successfully")

    # --------------------------------------------------
    # TEMP: Dummy feature builder (FAST + stable)
    # --------------------------------------------------
    def _build_features_from_bytes(self, video_bytes: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TEMPORARY for deployment stability:
        - Replace later with real pipeline:
          (1) decode video
          (2) pose keypoints per frame (T=16)
          (3) appearance features (512)
        """
        pose = torch.zeros((1, 16, 73), device=self.device, dtype=torch.float32)
        app = torch.zeros((1, 512), device=self.device, dtype=torch.float32)
        return pose, app

    # ----------------------------
    # Prediction API
    # ----------------------------
    @torch.no_grad()
    def predict(self, video_bytes: bytes) -> Dict:
        """
        Returns frontend-compatible keys to avoid 'Failed to fetch' due to mismatch.
        """
        pose, app = self._build_features_from_bytes(video_bytes)

        logits = self.model(pose, app)           # (1, C)
        probs = torch.softmax(logits, dim=1)[0]  # (C,)

        # Map model CLASS_NAMES to frontend 4 buckets
        # Your current UI expects exactly these 4 keys.
        buckets = ["emotion", "social", "physical", "pose_idle"]

        # Start all at 0
        class_probs = {k: 0.0 for k in buckets}

        # If your CLASS_NAMES already contains these exact labels, we can fill them.
        # Otherwise, fallback: pick top-1 and assign it to pose_idle.
        name_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

        any_matched = False
        for k in buckets:
            if k in name_to_idx:
                class_probs[k] = float(probs[name_to_idx[k]].item())
                any_matched = True

        if not any_matched:
            # fallback: send something valid to UI
            class_probs["pose_idle"] = 1.0

        # Normalize just in case (ensure sum ~ 1)
        s = sum(class_probs.values())
        if s > 0:
            for k in class_probs:
                class_probs[k] = class_probs[k] / s
        else:
            class_probs = {"emotion": 0.0, "social": 0.0, "physical": 0.0, "pose_idle": 1.0}

        # predicted class = argmax
        predicted_class = max(class_probs.items(), key=lambda kv: kv[1])[0]
        confidence = float(class_probs[predicted_class])

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probs
        }
