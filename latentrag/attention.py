import math

import torch
import torch.nn.functional as F


def inverse_self_attention(
    retrieved_latents: torch.Tensor,
    temperature: float = 1.0,
    reciprocal_epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Equation (3)--(5): inverse self-attention over each W+ layer.

    Args:
        retrieved_latents: [B, m, 14, 512] or [m, 14, 512].
    Returns:
        One augmented code per batch item with shape [B, 14, 512].
    """
    if retrieved_latents.ndim == 3:
        retrieved_latents = retrieved_latents.unsqueeze(0)
    if retrieved_latents.ndim != 4 or tuple(retrieved_latents.shape[2:]) != (14, 512):
        raise ValueError(f"expected [B,m,14,512], got {tuple(retrieved_latents.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    # [B,m,L,C] -> [B,L,m,C], so attention is computed independently per layer.
    values = retrieved_latents.permute(0, 2, 1, 3).contiguous()
    similarity = torch.matmul(values, values.transpose(-1, -2)) / math.sqrt(512.0)
    sign = torch.where(similarity < 0, -torch.ones_like(similarity), torch.ones_like(similarity))
    reciprocal = 1.0 / torch.where(similarity.abs() < reciprocal_epsilon, sign * reciprocal_epsilon, similarity)
    weights = F.softmax(reciprocal / temperature, dim=-1)
    fused = torch.matmul(weights, values)
    # Average the m retrieved codes, leaving one 512-D vector for every layer.
    return fused.mean(dim=2)
