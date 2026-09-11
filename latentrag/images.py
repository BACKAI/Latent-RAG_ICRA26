from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PathLike = Union[str, Path]


def image_paths(root: PathLike) -> List[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_rgb(path: PathLike) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def pil_to_tensor(image: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    """Convert an image to StyleGAN's [-1, 1] BCHW convention."""
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous() / 127.5 - 1.0
    return tensor.unsqueeze(0).to(device)


def path_to_generator_tensor(path: PathLike, size: int, device: torch.device) -> torch.Tensor:
    return pil_to_tensor(load_rgb(path), size, device)


def to_reid_input(images: torch.Tensor, height: int = 256, width: int = 128) -> torch.Tensor:
    """Resize StyleGAN-range images for the Market-1501 re-ID encoder."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB images, got {tuple(images.shape)}")
    images = (images + 1.0) / 2.0
    images = F.interpolate(images, size=(height, width), mode="bilinear", align_corners=False)
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images - mean) / std


def save_tensor_image(image: torch.Tensor, path: PathLike) -> None:
    """Save one image in a tensor batch or a single CHW tensor."""
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB image, got {tuple(image.shape)}")
    array = ((image.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5)
    array = array.permute(1, 2, 0).round().to(torch.uint8).numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)
