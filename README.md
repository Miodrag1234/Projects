# Deep Image Stitching — UDIS Extension & Panorama Pipeline

Portfolio repository for an **MSc / research project**: unsupervised panorama stitching built on [UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching) (Nie et al., IEEE TIP 2021), with **custom Net4 experiments**, **robustness evaluation**, custom dataset **FERIT-Panorama26**, and Stage 2 model **UDIS-V1**.

> **For recruiters:** everything below is viewable **without downloading datasets or weights** — sample images, metrics, and code entry points are in this repo.

---

## What you can see immediately

| What | Where |
|------|--------|
| **Stitched panorama (output)** | [`examples/stitched/panorama_000030.jpg`](examples/stitched/panorama_000030.jpg) |
| **More demo assets** | [`examples/`](examples/) — add input pairs / alignments (see `examples/README.md`) |
| **Quantitative results** | [Results summary](docs/RESULTS.md) (PSNR / SSIM tables) |
| **My code vs. original UDIS** | [Contributions](docs/CONTRIBUTIONS.md) |
| **Custom dataset (description only)** | [FERIT-Panorama26 README](datasets/FERIT-Panorama26/README.md) — **no full image dump in repo** |
| **Architecture overview** | [`network.jpg`](network.jpg) |
| **Reproducible environment** | [`environment.yml`](environment.yml) |

### Sample result (Stage 2 — UDIS-V1)

![Example stitched panorama](examples/stitched/panorama_000030.jpg)

---

## Highlights (CV bullet points)

- End-to-end **two-stage** deep stitching pipeline (alignment → reconstruction).
- **Architecture experiment:** added **Net4** refinement block; compared **joint fine-tuning** vs **frozen-backbone** training.
- **Evaluation:** PSNR / SSIM in overlap region; **multi-resolution robustness** script (`evaluate_robustness.py`).
- **Custom street-view dataset** **FERIT-Panorama26** (Mapillary) — documented with sample images only.
- **UDIS-V1:** trained Stage 2 reconstruction model (100k iterations); training metrics documented.
- **Engineering:** permissive checkpoint loading, flexible data loader (folder pairs + flat sequences), Windows + Conda setup.

Full attribution: [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md).

---

## Pipeline

```
Image pair  →  Stage 1: homography (UDIS-Base / Net4 variants)
            →  warped images + masks
            →  Stage 2: reconstruction (UDIS-V1)
            →  panorama
```

---

## Key scripts

### Stage 1 — `ImageAlignment/Codes/`

| Script | Purpose |
|--------|---------|
| [`train_H.py`](ImageAlignment/Codes/train_H.py) | Train / fine-tune alignment (Net4, `FREEZE_BACKBONE`) |
| [`inference.py`](ImageAlignment/Codes/inference.py) | PSNR / SSIM evaluation |
| [`evaluate_robustness.py`](ImageAlignment/Codes/evaluate_robustness.py) | Robustness across input resolutions |
| [`output_inference.py`](ImageAlignment/Codes/output_inference.py) | Export warps/masks for Stage 2 |

### Stage 2 — `ImageReconstruction/Codes/`

| Script | Purpose |
|--------|---------|
| [`train.py`](ImageReconstruction/Codes/train.py) | Train **UDIS-V1** reconstruction |
| [`inference.py`](ImageReconstruction/Codes/inference.py) | Generate final panoramas |

Configure paths in each `Codes/constant.py`.

---

## Model names (for comparing experiments)

| Name | Description |
|------|-------------|
| **UDIS-Base** | Official pretrained Stage 1 (1M iters, **H₃**) — best on UDIS-D |
| **UDIS-Net4-Joint** | Net4 + fine-tune all weights (1.04M iters, **H₄**) |
| **UDIS-Net4-Frozen** | Net4 only, frozen Net1–Net3 (1.2M iters, **H₄**) |
| **UDIS-V1** | Custom Stage 2 reconstruction checkpoint (e.g. 100k iters) |

### Results at a glance (UDIS-D test, 1105 pairs)

| Model | PSNR | SSIM |
|-------|------|------|
| **UDIS-Base** | **22.56** | **0.695** |
| UDIS-Net4-Joint | 21.38 | 0.660 |
| UDIS-Net4-Frozen | 19.97 | 0.623 |

More tables (robustness, FERIT-Panorama26): [docs/RESULTS.md](docs/RESULTS.md).

---

## Datasets in this repo

| Dataset | Included in GitHub? |
|---------|---------------------|
| **UDIS-D** | **No** — download from [UDIS project](https://github.com/nie-lang/UnsupervisedDeepImageStitching) |
| **FERIT-Panorama26** | **README + sample images only** — [datasets/FERIT-Panorama26/](datasets/FERIT-Panorama26/) |
| Pretrained **checkpoints** | **No** — links in original UDIS repo (Google Drive / Baidu) |
| **VGG-19** weights | **No** — generate locally via `ImageReconstruction/vgg19/convert_vgg19.py` |

---

## Quick start (evaluation only)

Requirements: **Python 3.6**, **TensorFlow 1.13.1**, GPU recommended (tested on GTX 1070 Ti).

```bash
conda env create -f environment.yml
conda activate udis-d
```

1. Download UDIS Stage 1 checkpoint (`model.ckpt-1000000`) → `ImageAlignment/Codes/checkpoints_homo/`
2. Download UDIS-D or use your own image pairs (see dataset READMEs)
3. Run Stage 1 metrics:

```bash
cd ImageAlignment/Codes
python inference.py
```

Optional Stage 2: `output_inference.py` → `ImageReconstruction/Codes/train.py` / `inference.py`.

---

## Repository layout

```
.
├── README.md
├── network.jpg
├── environment.yml
├── docs/
│   ├── ATTRIBUTION.md
│   ├── CONTRIBUTIONS.md
│   └── RESULTS.md
├── examples/                    # portfolio visuals (small)
├── datasets/
│   └── FERIT-Panorama26/        # dataset doc + sample images only
├── ImageAlignment/Codes/        # Stage 1
└── ImageReconstruction/Codes/   # Stage 2
```

Large folders (training images, checkpoints, full `results/`) are **gitignored** — see [`.gitignore`](.gitignore).

---

## Citation (original UDIS)

```bibtex
@article{nie2021udis,
  author  = {Nie, Lang and Lin, Chunyu and Liao, Kang and Liu, Shuaicheng and Zhao, Yao},
  title   = {Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images},
  journal = {IEEE Transactions on Image Processing},
  volume  = {30},
  pages   = {6184--6197},
  year    = {2021}
}
```

This repository: cite the GitHub URL and describe **your contributions** in [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md).

---

## Contact

**[Your Name]** · **[email]** · **[LinkedIn / portfolio URL]**

**Stack:** Python · TensorFlow 1.x · OpenCV · Computer Vision · Deep Learning · Image Stitching
