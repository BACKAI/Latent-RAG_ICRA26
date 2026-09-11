"""Latent-RAG: identity retrieval-guided latent augmentation."""

from .config import LatentRAGConfig
from .algorithm import LatentRAG

__all__ = ["LatentRAG", "LatentRAGConfig"]
