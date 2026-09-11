#!/usr/bin/env python3
"""Run frozen-generator Latent-RAG protection on query images."""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latentrag.algorithm import LatentRAG
from latentrag.config import LatentRAGConfig
from latentrag.generator import load_stylegan3
from latentrag.images import image_paths, path_to_generator_tensor, save_tensor_image
from latentrag.reid import load_reid_encoder
from latentrag.retrieval import LatentVectorStore


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One aligned image or a directory of aligned images")
    parser.add_argument("--output", required=True)
    parser.add_argument("--generator", required=True, help="StyleGAN3 *.pkl")
    parser.add_argument("--reid-checkpoint", required=True)
    parser.add_argument("--vector-store", required=True, help="Store made by build_vector_store.py")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--alpha-coarse", type=float, default=0.001)
    parser.add_argument("--alpha-fine", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--attention-temperature", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    generator = load_stylegan3(args.generator, device)
    reid = load_reid_encoder(args.reid_checkpoint, device)
    store = LatentVectorStore.load(args.vector_store)
    config = LatentRAGConfig(
        top_k=args.top_k,
        iterations=args.iterations,
        alpha_coarse=args.alpha_coarse,
        alpha_fine=args.alpha_fine,
        momentum=args.momentum,
        attention_temperature=args.attention_temperature,
    )
    method = LatentRAG(generator, reid, store, config, device)
    paths = image_paths(args.input)
    if not paths:
        raise RuntimeError("No input images were found")

    output = Path(args.output)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "latents").mkdir(parents=True, exist_ok=True)
    (output / "metadata").mkdir(parents=True, exist_ok=True)
    resolution = int(getattr(generator, "img_resolution", getattr(generator.synthesis, "img_resolution", 256)))

    for path in tqdm(paths, desc="Latent-RAG"):
        query = path_to_generator_tensor(path, resolution, device)
        protected, latent, metadata = method.protect(query)
        stem = path.stem
        save_tensor_image(protected, output / "images" / f"{stem}.png")
        torch.save(latent.cpu(), output / "latents" / f"{stem}.pt")
        metadata.update({"input": str(path), "output": str(output / "images" / f"{stem}.png")})
        with open(output / "metadata" / f"{stem}.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)


if __name__ == "__main__":
    main()
