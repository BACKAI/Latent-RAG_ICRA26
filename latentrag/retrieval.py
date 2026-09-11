from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _first(mapping: Dict, names: Sequence[str]):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


class LatentVectorStore:
    """Identity-vector to E4E-latent lookup table.

    The store is deliberately separate from query images.  This is the
    paper's privacy-critical decoupling: a query must not have its own latent
    in the retrieval database.
    """

    def __init__(self, features: torch.Tensor, latents: torch.Tensor, names=None, use_faiss: bool = True):
        if features.ndim != 2:
            raise ValueError(f"features must be [N,D], got {tuple(features.shape)}")
        if latents.ndim != 3 or latents.shape[0] != features.shape[0]:
            raise ValueError("latents must be [N,14,512] and align with features")
        if tuple(latents.shape[1:]) != (14, 512):
            raise ValueError(f"expected W+ shape [N,14,512], got {tuple(latents.shape)}")
        self.features = F.normalize(features.float().cpu(), dim=1).contiguous()
        self.latents = latents.float().cpu().contiguous()
        self.names = np.asarray(names if names is not None else [str(i) for i in range(len(latents))])
        if len(self.names) != len(latents):
            raise ValueError("names must contain one entry per latent")
        self._faiss_index = None
        if use_faiss:
            try:
                import faiss

                self._faiss_index = faiss.IndexFlatIP(self.features.shape[1])
                self._faiss_index.add(self.features.numpy())
            except ImportError:
                pass

    @classmethod
    def load(cls, path: str, use_faiss: bool = True) -> "LatentVectorStore":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vector store does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix == ".npz":
            data = dict(np.load(path, allow_pickle=True))
        elif suffix == ".mat":
            from scipy.io import loadmat

            data = loadmat(path)
        elif suffix in {".pt", ".pth"}:
            data = torch.load(path, map_location="cpu")
            if not isinstance(data, dict):
                raise ValueError("A .pt vector store must be a dictionary")
        else:
            raise ValueError("Vector store must be .npz, .mat, .pt, or .pth")

        features = _first(data, ("features", "gallery_f", "gallery_features", "identity_vectors"))
        latents = _first(data, ("latents", "gallery_e4e", "gallery_latents", "w_plus"))
        names = _first(data, ("names", "paths", "gallery_names"))
        if features is None or latents is None:
            raise KeyError("Store must contain features/gallery_f and latents/gallery_e4e")
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(np.asarray(features))
        if not isinstance(latents, torch.Tensor):
            latents = torch.as_tensor(np.asarray(latents))
        if features.ndim == 2 and features.shape[0] == 512 and features.shape[1] != 512:
            features = features.t()
        if latents.ndim == 2 and latents.shape[1] == 14 * 512:
            latents = latents.reshape(latents.shape[0], 14, 512)
        elif latents.ndim == 2 and latents.shape[0] == 14 * 512:
            latents = latents.t().reshape(latents.shape[1], 14, 512)
        elif latents.ndim == 3 and latents.shape[0] == 14 and latents.shape[2] == 512:
            latents = latents.permute(1, 0, 2)
        if names is not None:
            names = [str(x) for x in np.asarray(names).reshape(-1)]
        return cls(features, latents, names=names, use_faiss=use_faiss)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            features=self.features.numpy().astype(np.float32),
            latents=self.latents.numpy().astype(np.float32),
            names=self.names,
        )

    def search(self, query_features: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if query_features.ndim == 1:
            query_features = query_features.unsqueeze(0)
        if query_features.ndim != 2 or query_features.shape[1] != self.features.shape[1]:
            raise ValueError("query feature dimension does not match the vector store")
        k = min(int(k), len(self.latents))
        if k < 1:
            raise ValueError("the vector store is empty")
        query = F.normalize(query_features.float().cpu(), dim=1).contiguous()
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query.numpy(), k)
            return torch.from_numpy(indices).long(), torch.from_numpy(scores).float()
        scores = query @ self.features.t()
        scores, indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
        return indices, scores

    def latents_for(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        return self.latents[indices.cpu()].to(device)
