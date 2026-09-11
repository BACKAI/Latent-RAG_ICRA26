# Latent-RAG: Identity Retrieval-Guided Latent Augmentation

This directory contains an independent implementation of **Latent-RAG: Identity Retrieval-Guided Latent Augmentation for Privacy-Preserving Person Re-Identification**, an **accepted paper at the IEEE International Conference on Robotics and Automation (ICRA 2026), Vienna, Austria**. The official proceedings version was not yet available when this implementation was prepared; the code follows the supplied final-paper PDF.

The original repository at `../Latent-RAG_ICRA26-main` was inspected for model/checkpoint conventions but is intentionally not modified. This directory uses the paper's stated StyleGAN3 generator and implements the retrieval, inverse self-attention, and identity-aligned visual-divergence stages independently.

## What the method does

Latent-RAG protects a person image while retaining a re-identification embedding:

1. A frozen Market-1501 re-ID encoder extracts a 512-D identity vector from the query.
2. FAISS retrieves the top `m=10` identity vectors from a precomputed gallery vector store. The query itself must not be present in this store.
3. The paired E4E `W+` codes are fused with inverse self-attention. For each of the 14 StyleGAN layers, the paper computes `S = (QK^T)/sqrt(512)`, replaces it with the element-wise reciprocal, applies row-wise softmax, and averages the retrieved codes.
4. The resulting `w*` is updated for `T=10` iterations. Layers 0--2 use the visual-divergence loss `Ldiv = -||G(w*) - I||` with step size `0.001`; layers 3--13 use the identity loss `Lid = 1 - cosine(E(G(w*)), f)` with step size `0.01`. Momentum is `0.9` and the updates use the iterative fast-gradient-sign rule described in the paper.
5. The frozen StyleGAN3 generator synthesizes the protected image.

No generator, re-ID, or E4E weights are trained by Latent-RAG. The only per-image optimization is the ten-step latent update above. This is important: the method is intended to work without additional privacy-model training.

## Repository layout

```text
latentrag/                  Core algorithm, loaders, attention, retrieval, and image utilities
models/e4e_reid/            E4E encoder building blocks used to build the offline store
scripts/build_vector_store.py
                            Offline Market-1501 identity/latent store construction
scripts/inference.py        Query-image protection
third_party/stylegan3/      Official NVlabs StyleGAN3 source used to load network pickles
```

## Environment

Use a CUDA-matched PyTorch and torchvision installation. The reported experiment used one NVIDIA RTX 3080 GPU. A CPU fallback is supported for smoke tests, but full StyleGAN3 inference and store construction are intended for CUDA.

```bash
cd Latent-RAG_real
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the CUDA-matched torch/torchvision pair for your system first.
python -m pip install -r requirements.txt
```

`faiss-cpu` is sufficient for a single-machine vector store. Use a compatible FAISS GPU build if desired; the code automatically uses FAISS when available and otherwise falls back to exact PyTorch inner-product search.

## Required assets

Place the following files anywhere on disk and pass their paths to the scripts:

| Asset | Purpose |
| --- | --- |
| StyleGAN3 FFHQ pickle, preferably `stylegan3-t-ffhqu-256x256.pkl` (officially 16 W slots with 512-D styles) | Frozen generator `G` |
| E4E checkpoint configured for a 256px StyleGAN latent space | Produces the 14 x 512 `W+` code for each gallery image |
| Market-1501-trained ResNet-50 re-ID checkpoint | Query/gallery identity vectors and `Lid` |

The vendored official StyleGAN3 code documents the official NVIDIA checkpoint names and links. The re-ID and E4E checkpoints should be the same pretrained weights used to create the vector store; mixing encoder versions invalidates the stored identity/latent pairs. The loader accepts the checkpoint layouts used by the inspected reference implementation, including `module.` and `state_dict` prefixes.

The generator loader accepts the paper's 14-slot interface and official StyleGAN3 checkpoints that expose 16 internal slots (14 synthesis layers plus Fourier-input and ToRGB slots). For a 16-slot checkpoint, the wrapper reuses the boundary W+ codes for those two architectural slots. Any other `num_ws` or a `w_dim` other than 512 is rejected.

## Dataset and vector-store construction

The paper evaluates on **Market-1501**, **MSMT17**, and **CUHK03**. The reported vector store is built from `N=32,668` Market-1501 gallery samples. The standard way to obtain that count is to use the `bounding_box_train` and `bounding_box_test` directories and exclude `query`; verify the exact file list for your copy of the dataset.

The store contains two aligned arrays:

* `features`: normalized 512-D Market-1501 identity vectors (`gallery_f` in the reference `.mat` convention).
* `latents`: E4E `W+` codes with shape `[N, 14, 512]` (`gallery_e4e` in the reference `.mat` convention).

Build it once on the secure server:

```bash
python scripts/build_vector_store.py \
  --gallery-dir /data/Market-1501/bounding_box_train \
  --gallery-dir /data/Market-1501/bounding_box_test \
  --reid-checkpoint /models/resnet50_reid.pth \
  --e4e-checkpoint /models/e4e_reid.pt \
  --output /secure_store/market1501_gallery.npz \
  --device cuda \
  --batch-size 8
```

Do not add query images to `market1501_gallery.npz`. The store can also load the reference project's `vector_store.mat` when it contains `gallery_f` and `gallery_e4e`; `.npz` is the recommended format for newly built stores. The store is about 1 GB in float32 at the paper's 32,668-sample size, so keep it on fast local storage and do not transmit it to an untrusted client.

The script is an offline preparation stage, not additional Latent-RAG training. It uses the frozen E4E and re-ID encoders in evaluation mode and writes one paired entry per gallery image.

## Inference

Input images should be RGB, aligned/cropped person images. The implementation resizes each image to the generator's square resolution for StyleGAN3 and separately resizes it to `256 x 128` with ImageNet normalization for the Market-1501 re-ID encoder.

```bash
python scripts/inference.py \
  --input /data/Market-1501/query \
  --output outputs/market1501_query \
  --generator /models/stylegan3-t-ffhqu-256x256.pkl \
  --reid-checkpoint /models/resnet50_reid.pth \
  --vector-store /secure_store/market1501_gallery.npz \
  --device cuda
```

Outputs are written to:

```text
outputs/market1501_query/
├── images/       protected PNG images
├── latents/      final 14 x 512 adversarial W+ tensors
└── metadata/     retrieved indices/scores and per-step losses
```

The defaults are the paper settings: `top_k=10`, `iterations=10`, `alpha_coarse=0.001`, `alpha_fine=0.01`, `momentum=0.9`, and inverse-attention temperature `1.0` (the temperature is not an additional trainable parameter). All options can be overridden from the command line for ablations, but reported reproduction runs should use the defaults.

## Evaluation protocol

The paper reports:

* privacy: PSNR and SSIM between original/protected and original/recovered images, where lower is better;
* utility: Rank-1 and mAP using AGW, BagTricks, and ABD-Net re-ID backbones;
* recovery robustness: a black-box attacker collects input/protected pairs and trains a Restormer recovery network with L1 reconstruction loss.

For a faithful benchmark, protect only the query split, keep the raw gallery images and their identity labels unchanged, and evaluate each protected query against the raw gallery embeddings. Train the recovery attacker only on the public paired data allowed by the threat model, using the same dataset split policy for every privacy method. Restormer is intentionally not bundled here because it is an evaluation-time attacker, not a component of Latent-RAG; its output can be evaluated by the same PSNR/SSIM code used for protected outputs.

The paper's main reported values are reproduced below as a reference target, not as a claim that a run has already been executed in this environment:

| Dataset | Protected PSNR | Protected SSIM | Recovery PSNR | Recovery SSIM |
| --- | ---: | ---: | ---: | ---: |
| Market-1501 | 10.43 | 0.07 | 14.14 | 0.22 |
| MSMT17 | 12.17 | 0.11 | 15.73 | 0.25 |
| CUHK03 | 9.17 | 0.06 | 12.91 | 0.18 |

The paper reports Market-1501 Rank-1/mAP of `94.4/87.1` for AGW, `93.9/85.7` for BagTricks, and `95.1/88.9` for ABD-Net. Exact numbers depend on the released pretrained checkpoints, image lists, alignment, and evaluation code.

## Implementation notes and reproducibility choices

* The protected query is never encoded by E4E at inference; only the retrieved gallery W+ codes are used. This preserves the paper's input/latent decoupling.
* Inverse attention is implemented layer-wise as `[B,14,m,512]`, with `Q=K=V` equal to each layer's retrieved code matrix. Reciprocal scores are stabilized only at values numerically indistinguishable from zero.
* `Ldiv` is negative Euclidean image distance, exactly matching Eq. (6); gradient descent therefore increases visual distance. `Lid` is one minus cosine similarity, matching Eq. (7).
* Re-ID and StyleGAN3 parameters are frozen. Gradients are taken only with respect to the current coarse or fine latent slice.
* The implementation uses the official NVlabs StyleGAN3 loader under `third_party/stylegan3`. That directory is an upstream dependency and retains its upstream license.
* The supplied older code uses StyleGAN2-ADA naming, hard-coded CUDA calls, an incomplete dependency set, and an inverse-attention temperature not stated in the paper. Those issues are not copied into this implementation.

## Citation

```bibtex
@inproceedings{back2026latentrag,
  title     = {Latent-RAG: Identity Retrieval-Guided Latent Augmentation for Privacy-Preserving Person Re-Identification},
  author    = {Back, Seung-hyeok and Lee, Eungi and Kim, Hyung-Il and Yoo, Seok Bong},
  booktitle = {2026 IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026},
  address   = {Vienna, Austria},
  note      = {Accepted; proceedings version forthcoming}
}
```
