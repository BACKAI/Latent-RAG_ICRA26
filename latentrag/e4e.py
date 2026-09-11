from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

from models.e4e_reid.encoders import psp_encoders


def _encoder_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("E4E checkpoint has no state dictionary")
    result = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("encoder."):
            result[key[len("encoder."):]] = value
        else:
            result[key] = value
    return result


class E4EEncoder(torch.nn.Module):
    """E4E W+ encoder, restricted to the paper's 14-layer output."""

    def __init__(self, checkpoint: str, device: torch.device, stylegan_size: int = 256):
        super().__init__()
        payload = torch.load(Path(checkpoint), map_location="cpu")
        opts = dict(payload.get("opts", {})) if isinstance(payload, dict) else {}
        # E4E computes 2*log2(size)-2 styles.  256 -> 14, as required here.
        opts["stylegan_size"] = stylegan_size
        options = Namespace(**opts)
        encoder_type = getattr(options, "encoder_type", "Encoder4Editing")
        if encoder_type == "GradualStyleEncoder":
            encoder = psp_encoders.GradualStyleEncoder(50, "ir_se", options)
        elif encoder_type == "SingleStyleCodeEncoder":
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, "ir_se", options)
        else:
            encoder = psp_encoders.Encoder4Editing(50, "ir_se", options)
        expected = encoder.state_dict()
        compatible = {
            key: value for key, value in _encoder_state_dict(payload).items()
            if key in expected and expected[key].shape == value.shape
        }
        if not compatible:
            raise RuntimeError(f"No compatible tensors found in E4E checkpoint: {checkpoint}")
        encoder.load_state_dict(compatible, strict=False)
        self.encoder = encoder.to(device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        print(f"Loaded {len(compatible)} E4E tensors from {checkpoint}")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        latents = self.encoder(images)
        if latents.ndim != 3 or tuple(latents.shape[1:]) != (14, 512):
            raise ValueError(f"E4E output must be [B,14,512], got {tuple(latents.shape)}")
        return latents
