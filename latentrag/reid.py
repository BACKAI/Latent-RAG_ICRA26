from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ClassBlock(nn.Module):
    """Compatible with the re-ID checkpoint layout used by the reference code."""

    def __init__(self, input_dim: int = 2048, class_num: int = 751, linear: int = 512):
        super().__init__()
        self.add_block = nn.Sequential(nn.Linear(input_dim, linear), nn.BatchNorm1d(linear))
        self.classifier = nn.Sequential(nn.Linear(linear, class_num))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.add_block(x))


class MarketReIDEncoder(nn.Module):
    """ResNet-50 + 512-D embedding used for Market-1501 retrieval and loss."""

    def __init__(self, class_num: int = 751):
        super().__init__()
        try:
            backbone = models.resnet50(weights=None)
        except TypeError:  # torchvision < 0.13
            backbone = models.resnet50(pretrained=False)
        backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = backbone
        self.classifier = ClassBlock(2048, class_num=class_num, linear=512)

    def _backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(images)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return x.flatten(1)

    def embedding(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.classifier.add_block(self._backbone_features(images)), dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._backbone_features(images))


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "net"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise ValueError("Re-ID checkpoint does not contain a state dictionary")
    result = {}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(key)
        result[key[7:] if key.startswith("module.") else key] = value
    return result


def load_reid_encoder(checkpoint: Optional[str], device: torch.device, class_num: int = 751) -> MarketReIDEncoder:
    model = MarketReIDEncoder(class_num=class_num)
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        payload = torch.load(checkpoint_path, map_location="cpu")
        raw = _extract_state_dict(payload)
        current = model.state_dict()
        compatible = {k: v for k, v in raw.items() if k in current and current[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        if not compatible:
            raise RuntimeError(f"No compatible tensors found in Re-ID checkpoint: {checkpoint_path}")
        print(f"Loaded {len(compatible)} Re-ID tensors from {checkpoint_path}")
        if missing:
            print(f"Warning: {len(missing)} Re-ID tensors were not present in the checkpoint")
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected Re-ID tensors were ignored")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
