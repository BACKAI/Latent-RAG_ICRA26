from dataclasses import dataclass


@dataclass
class LatentRAGConfig:
    """Hyperparameters reported in the ICRA 2026 paper.

    The generator and the E4E checkpoint are expected to expose a 14 x 512
    W+ latent.  ``coarse_layers`` is three because StyleGAN's layers 0--2
    encode coarse structure and layers 3--13 encode finer attributes.
    """

    top_k: int = 10
    latent_layers: int = 14
    latent_dim: int = 512
    coarse_layers: int = 3
    iterations: int = 10
    alpha_coarse: float = 0.001
    alpha_fine: float = 0.01
    momentum: float = 0.9
    attention_temperature: float = 1.0
    reciprocal_epsilon: float = 1.0e-6
    reid_height: int = 256
    reid_width: int = 128

    def validate(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.latent_layers != 14 or self.latent_dim != 512:
            raise ValueError("The paper uses W+ latents with shape 14 x 512")
        if not 0 < self.coarse_layers < self.latent_layers:
            raise ValueError("coarse_layers must split the 14 W+ layers")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.alpha_coarse <= 0 or self.alpha_fine <= 0:
            raise ValueError("latent update step sizes must be positive")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if self.attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
