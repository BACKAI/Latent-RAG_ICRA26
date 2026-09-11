import sys
from pathlib import Path
from typing import Any

import torch


def _stylegan3_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "stylegan3"


def _num_ws(generator: torch.nn.Module) -> int:
    if hasattr(generator, "num_ws"):
        return int(generator.num_ws)
    if hasattr(generator, "synthesis") and hasattr(generator.synthesis, "num_ws"):
        return int(generator.synthesis.num_ws)
    return -1


def load_stylegan3(path: str, device: torch.device) -> torch.nn.Module:
    """Load a StyleGAN3 network pickle with the official NVlabs loader."""
    root = _stylegan3_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import legacy
    except ImportError as exc:
        raise RuntimeError(f"Cannot import vendored StyleGAN3 loader from {root}") from exc

    with open(path, "rb") as file:
        networks = legacy.load_network_pkl(file)
    if "G_ema" not in networks:
        raise KeyError("StyleGAN3 pickle does not contain G_ema")
    generator = networks["G_ema"].to(device).eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    num_ws = _num_ws(generator)
    if num_ws not in (14, 16):
        raise ValueError(f"Expected a 14-slot paper generator or official StyleGAN3 16-slot generator, got num_ws={num_ws}")
    if int(getattr(generator, "w_dim", 512)) != 512:
        raise ValueError("The paper requires a 512-dimensional W space")
    return generator


def to_generator_w_plus(generator: torch.nn.Module, w_plus: torch.Tensor) -> torch.Tensor:
    num_ws = _num_ws(generator)
    if w_plus.ndim != 3 or w_plus.shape[2] != 512:
        raise ValueError(f"expected [B,14,512] W+ input, got {tuple(w_plus.shape)}")
    if w_plus.shape[1] == num_ws:
        return w_plus
    if w_plus.shape[1] == 14 and num_ws == 16:
        return torch.cat([w_plus[:, :1], w_plus, w_plus[:, -1:]], dim=1)
    raise ValueError(f"cannot map W+ with {w_plus.shape[1]} slots to generator with {num_ws} slots")


def synthesize(generator: torch.nn.Module, w_plus: torch.Tensor) -> torch.Tensor:
    """Synthesize with deterministic StyleGAN noise, preserving input gradients."""
    w_plus = to_generator_w_plus(generator, w_plus)
    try:
        return generator.synthesis(w_plus, noise_mode="const", force_fp32=True)
    except TypeError:
        return generator.synthesis(w_plus, noise_mode="const")
