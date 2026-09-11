from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .attention import inverse_self_attention
from .config import LatentRAGConfig
from .generator import synthesize
from .images import to_reid_input
from .retrieval import LatentVectorStore


class LatentRAG:
    """Frozen-generator Latent-RAG inference.

    There is intentionally no optimizer over generator/re-ID parameters.  The
    only iterative operation is the ten-step latent update described in the
    paper.
    """

    def __init__(self, generator, reid_encoder, vector_store: LatentVectorStore, config: LatentRAGConfig, device):
        config.validate()
        self.generator = generator
        self.reid = reid_encoder
        self.vector_store = vector_store
        self.config = config
        self.device = torch.device(device)

    def _identity(self, image: torch.Tensor) -> torch.Tensor:
        return self.reid.embedding(to_reid_input(image, self.config.reid_height, self.config.reid_width))

    def _update_latent(self, w_star: torch.Tensor, query_image: torch.Tensor, query_identity: torch.Tensor):
        coarse = w_star[:, : self.config.coarse_layers].detach()
        fine = w_star[:, self.config.coarse_layers :].detach()
        coarse_momentum = torch.zeros_like(coarse)
        fine_momentum = torch.zeros_like(fine)
        history = []

        for step in range(self.config.iterations):
            coarse_var = coarse.detach().requires_grad_(True)
            full = torch.cat([coarse_var, fine], dim=1)
            generated = synthesize(self.generator, full)
            # Eq. (6): negative distance, updated by gradient descent to increase distance.
            divergence_loss = -torch.linalg.vector_norm(generated - query_image, ord=2, dim=(1, 2, 3)).mean()
            coarse_grad = torch.autograd.grad(divergence_loss, coarse_var, only_inputs=True)[0]
            coarse_grad = coarse_grad / coarse_grad.abs().flatten(1).mean(dim=1).clamp_min(1.0e-12).view(-1, 1, 1)
            coarse_momentum = self.config.momentum * coarse_momentum + coarse_grad
            coarse = (coarse - self.config.alpha_coarse * coarse_momentum.sign()).detach()

            fine_var = fine.detach().requires_grad_(True)
            full = torch.cat([coarse, fine_var], dim=1)
            generated = synthesize(self.generator, full)
            generated_identity = self._identity(generated)
            identity_loss = (1.0 - F.cosine_similarity(generated_identity, query_identity, dim=1)).mean()
            fine_grad = torch.autograd.grad(identity_loss, fine_var, only_inputs=True)[0]
            fine_grad = fine_grad / fine_grad.abs().flatten(1).mean(dim=1).clamp_min(1.0e-12).view(-1, 1, 1)
            fine_momentum = self.config.momentum * fine_momentum + fine_grad
            fine = (fine - self.config.alpha_fine * fine_momentum.sign()).detach()

            history.append({
                "step": step + 1,
                "divergence_loss": float(divergence_loss.detach().cpu()),
                "identity_loss": float(identity_loss.detach().cpu()),
            })

        return torch.cat([coarse, fine], dim=1), history

    def protect(self, query_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Protect a single BCHW query image in the generator's [-1,1] range."""
        if query_image.ndim != 4 or query_image.shape[0] != 1:
            raise ValueError("protect expects one image with shape [1,3,H,W]")
        query_image = query_image.to(self.device)
        with torch.no_grad():
            query_identity = self._identity(query_image)
            indices, scores = self.vector_store.search(query_identity.detach(), self.config.top_k)
            retrieved = self.vector_store.latents_for(indices[0], self.device).unsqueeze(0)
            w_star = inverse_self_attention(
                retrieved,
                temperature=self.config.attention_temperature,
                reciprocal_epsilon=self.config.reciprocal_epsilon,
            )

        w_adv, history = self._update_latent(w_star, query_image, query_identity.detach())
        with torch.no_grad():
            protected = synthesize(self.generator, w_adv)
        metadata = {
            "retrieved_indices": indices[0].tolist(),
            "retrieved_scores": scores[0].tolist(),
            "config": self.config.__dict__.copy(),
            "steps": history,
        }
        return protected.detach(), w_adv.detach(), metadata
