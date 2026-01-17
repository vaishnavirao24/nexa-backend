# src/inference.py
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Union, Optional

from .labels import CLASS_NAMES


# ----------------------------
#  Device
# ----------------------------
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
#  Pose Encoder (Transformer)
# ----------------------------
class PoseEncoder(nn.Module):
    def __init__(self, input_dim=73, d_model=128, nhead=4, num_layers=2):
        super().__init__()

        self.proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        # x: (B, T, 73)
        x = self.proj(x)
        x = self.encoder(x)
        return x.mean(dim=1)


# ----------------------------
#  Appearance Encoder
# ----------------------------
class AppearanceEncoder(nn.Module):
    def __init__(self, input_dim=512, out_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
#  Fusion Model
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

    def forward(self, pose, app):
        pose_feat = self.pose(pose)
        app_feat = self.app(app)

        fused = torch.cat([pose_feat, app_feat], dim=1)
        return self.classifier(fused)


# ----------------------------
#  Inference Wrapper
# ----------------------------
class NexaInference:
    """
    Render-safe inference wrapper.

    Fixes:
    - Accepts yolo_pose_path + device + extra kwargs (won't crash if app passes them)
    - predict() accepts bytes (FastAPI UploadFile)
    - Returns frontend UI schema:
        predicted_class, confidence, class_probabilities
    """

    def __init__(
        self,
        weights_path: Union[str, Path, None] = None,
        yolo_pose_path: Union[str, Path, None] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        # Accept and ignore extra kwargs to prevent future init crashes
        # e.g. kwargs may include yolo_model_path, config, etc.
        _ = kwargs

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = DEFAULT_DEVICE

        self.model = FusionModel(num_classes=len(CLASS_NAMES)).to(self.device)
        self.model.eval()

        # optional (kept for future; not used yet)
        self.yolo_pose_path: Optional[Path] = Path(yolo_pose_path) if yolo_pose_path else None

        # load weights if provided
        if weights_path is not None:
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
    #  Dummy feature loader (replace later with real pipeline)
    # --------------------------------------------------
    def _load_features_from_bytes(self, video_bytes: bytes):
        """
        TEMPORARY:
        Replace with real pipeline:
        - decode video bytes -> frames
        - pose extraction
        - resnet appearance features

        For now: zeros so end-to-end integration works.
        """
        pose = torch.zeros((1, 16, 73), device=self.device)
        app = torch.zeros((1, 512), device=self.device)
        return pose, app

    # ----------------------------
    #  Prediction API (bytes-in)
    # ----------------------------
    @torch.no_grad()
    def predict(self, video_bytes: bytes) -> Dict[str, Any]:
        pose, app = self._load_features_from_bytes(video_bytes)

        logits = self.model(pose, app)
        probs = torch.softmax(logits, dim=1)[0]  # (C,)

        # UI expects these exact keys
        ui_keys = ["emotion", "social", "physical", "pose_idle"]
        class_probabilities = {k: 0.0 for k in ui_keys}

        # Fill only if CLASS_NAMES match these keys
        for i, name in enumerate(CLASS_NAMES):
            if name in class_probabilities:
                class_probabilities[name] = float(probs[i].item())

        predicted_class = max(class_probabilities, key=class_probabilities.get)
        confidence = float(class_probabilities[predicted_class])

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
        }
