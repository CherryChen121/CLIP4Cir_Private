import os
from typing import List, Union

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from iden_modules.modeling.LoraRETFound import RETFound


class RETFoundAdapter(nn.Module):
    """
    Bridge RETFound vision backbone to CLIP4Cir's expected CLIP-like interface.

    - Image encoder: RETFound ViT-L/16 (pure vision)
    - Text encoder: OpenAI CLIP text tower
    - Unified output space: projection_dim
    """

    def __init__(
        self,
        backbone_path: str,
        text_model_name: str = "ViT-L/14",
        projection_dim: int = 768,
        input_resolution: int = 224,
    ):
        super().__init__()

        if not backbone_path:
            raise ValueError("RETFoundAdapter requires a valid backbone_path")
        if not os.path.exists(backbone_path):
            raise FileNotFoundError(f"RETFound backbone not found: {backbone_path}")

        self.vision_model = RETFound(pretrained=False)
        self._load_retfound_backbone(backbone_path)

        text_clip, _ = clip.load(text_model_name, device="cpu", jit=False)
        self.text_model = text_clip

        self.visual = self.vision_model
        self.visual.input_resolution = input_resolution
        self.visual.output_dim = projection_dim

        image_dim = 1024
        text_dim = int(self.text_model.text_projection.shape[1])

        self.image_projection = nn.Identity() if image_dim == projection_dim else nn.Linear(image_dim, projection_dim)
        self.text_projection = nn.Identity() if text_dim == projection_dim else nn.Linear(text_dim, projection_dim)

        # Keep CLIP-style initialization for compatibility.
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def _load_retfound_backbone(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint

        if isinstance(checkpoint, dict):
            if "model" in checkpoint and isinstance(checkpoint["model"], dict):
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]

        cleaned = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key.replace("module.", "", 1)
            if new_key.startswith("visual."):
                new_key = new_key.replace("visual.", "", 1)
            cleaned[new_key] = value

        missing, unexpected = self.vision_model.load_state_dict(cleaned, strict=False)
        print(
            f"RETFound backbone loaded from {checkpoint_path}. "
            f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        features = self.vision_model(image)
        features = self.image_projection(features)
        return F.normalize(features, dim=-1)

    def encode_text(self, text: Union[List[str], torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(text, list):
            text = clip.tokenize(text, context_length=77, truncate=True)
        text = text.to(device, non_blocking=True)
        features = self.text_model.encode_text(text)
        features = self.text_projection(features)
        return F.normalize(features, dim=-1)

    def forward(self, x, mode=None):
        if mode == "image":
            return self.encode_image(x)
        if mode == "text":
            return self.encode_text(x)
        raise ValueError("RETFoundAdapter.forward requires mode in ['image', 'text']")