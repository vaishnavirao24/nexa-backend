# src/inference.py
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from .labels import CLASS_NAMES


# ----------------------------
#  Device
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, weights_path: str | Path):
        self.device = DEVICE
        self.model = FusionModel(num_classes=len(CLASS_NAMES)).to(self.device)
        self.model.eval()

        self._load_weights(weights_path)

    def _load_weights(self, weights_path):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=self.device)

        # Support both raw state_dict and wrapped checkpoint
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        missing, unexpected = self.model.load_state_dict(
            state_dict,
            strict=False
        )

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
    #  Dummy feature loader (replace with real pipeline)
    # --------------------------------------------------
    def _load_features(self, path: str):
        """
        TEMPORARY:
        Replace this with:
        - pose extraction
        - resnet appearance features
        """

        # Fake inputs for now (to keep server alive)
        pose = torch.zeros((1, 16, 73), device=self.device)
        app = torch.zeros((1, 512), device=self.device)

        return pose, app

    # ----------------------------
    #  Prediction API
    # ----------------------------
    @torch.no_grad()
    def predict(self, video_path: str, top_k: int = 3):
        pose, app = self._load_features(video_path)

        logits = self.model(pose, app)
        probs = torch.softmax(logits, dim=1)[0]

        topk = torch.topk(probs, k=min(top_k, len(CLASS_NAMES)))

        result = {
            "predicted_label": CLASS_NAMES[topk.indices[0].item()],
            "confidence": float(topk.values[0].item()),
            "topk": [
                {
                    "label": CLASS_NAMES[idx.item()],
                    "score": float(score.item())
                }
                for idx, score in zip(topk.indices, topk.values)
            ]
        }

        return result
