# Panorama stitching (UDIS extension)

Deep learning pipeline for **panoramic image stitching**, based on [UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching) (Nie et al., IEEE TIP 2021), with **Net4 experiments**, **robustness evaluation**, dataset **FERIT-Panorama26**, and Stage 2 model **UDIS-V1**.

[← Back to portfolio root](../README.md)

---

## What you can see immediately

| What | Where |
|------|--------|
| **Stitched panorama** | [`examples/stitched/panorama_000030.jpg`](examples/stitched/panorama_000030.jpg) |
| **Results (PSNR / SSIM)** | [docs/RESULTS.md](docs/RESULTS.md) |
| **My contributions** | [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md) |
| **Dataset (docs only)** | [datasets/FERIT-Panorama26/](datasets/FERIT-Panorama26/) |

![Example panorama](examples/stitched/panorama_000030.jpg)

---

## Quick start

```bash
cd panorama-stitching
conda env create -f environment.yml
conda activate udis-d
cd ImageAlignment/Codes
python inference.py
```

See full details below.

---

## Pipeline

```
Image pair  →  Stage 1: homography (UDIS-Base / Net4)
            →  warps + masks
            →  Stage 2: UDIS-V1 reconstruction
            →  panorama
```

---

## Key scripts

| Stage | Folder | Scripts |
|-------|--------|---------|
| 1 | `ImageAlignment/Codes/` | `train_H.py`, `inference.py`, `evaluate_robustness.py`, `output_inference.py` |
| 2 | `ImageReconstruction/Codes/` | `train.py`, `inference.py` |

---

## Models

| Name | Description |
|------|-------------|
| **UDIS-Base** | Pretrained Stage 1 (H₃) — best on UDIS-D |
| **UDIS-Net4-Joint** | Net4 + joint fine-tune (H₄) |
| **UDIS-Net4-Frozen** | Net4-only fine-tune (H₄) |
| **UDIS-V1** | Custom Stage 2 reconstruction |

**UDIS-D (1105 pairs):** UDIS-Base **22.56 PSNR** / **0.695 SSIM** — see [docs/RESULTS.md](docs/RESULTS.md).

---

## Layout

```
panorama-stitching/
├── ImageAlignment/
├── ImageReconstruction/
├── datasets/
├── docs/
├── examples/
├── scripts/
├── environment.yml
└── network.jpg
```

Attribution: [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md).
