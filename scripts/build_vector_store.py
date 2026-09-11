#!/usr/bin/env python3
"""Build the identity/latent vector store used by Latent-RAG."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latentrag.e4e import E4EEncoder
from latentrag.images import image_paths, load_rgb, pil_to_tensor, to_reid_input
from latentrag.reid import load_reid_encoder
from latentrag.retrieval import LatentVectorStore


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-dir", action="append", required=True,
                        help="Gallery directory; repeat for Market train and test-gallery directories.")
    parser.add_argument("--reid-checkpoint", required=True)
    parser.add_argument("--e4e-checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Output .npz vector store")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stylegan-size", type=int, default=256,
                        help="E4E input size; 256 gives the paper's 14 W+ layers")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    paths = []
    for directory in args.gallery_dir:
        paths.extend(image_paths(directory))
    paths = sorted(set(paths))
    if args.max_images is not None:
        paths = paths[:args.max_images]
    if not paths:
        raise RuntimeError("No images were found in the supplied gallery directories")
    print(f"Building a vector store from {len(paths)} gallery images on {device}")

    reid = load_reid_encoder(args.reid_checkpoint, device)
    e4e = E4EEncoder(args.e4e_checkpoint, device, stylegan_size=args.stylegan_size)
    all_features = []
    all_latents = []
    all_names = []

    with torch.no_grad():
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start:start + args.batch_size]
            batch = torch.cat([pil_to_tensor(load_rgb(path), args.stylegan_size, device) for path in batch_paths])
            features = reid.embedding(to_reid_input(batch))
            latents = e4e(batch)
            all_features.append(features.cpu())
            all_latents.append(latents.cpu())
            all_names.extend(str(path) for path in batch_paths)
            print(f"[{min(start + args.batch_size, len(paths)):>{len(str(len(paths)))}d}/{len(paths)}]", end="\r")

    store = LatentVectorStore(torch.cat(all_features), torch.cat(all_latents), names=all_names)
    store.save(args.output)
    print(f"\nSaved {len(all_names)} entries to {args.output}")


if __name__ == "__main__":
    main()
