# My contributions (vs. original UDIS)

Items below were **added or substantially modified** in this project — not part of the upstream UDIS repository.

## Stage 1 — alignment (`ImageAlignment/`)

| File / feature | Description |
|----------------|-------------|
| `H_model.py` | **Net4** block and **H₄** output |
| `train_H.py` | `permissive_load`, `FREEZE_BACKBONE`, training progress bar, periodic checkpoints |
| `inference.py` | Permissive loading; evaluation modes (UDIS-Base vs Net4 variants); H₃ vs H₄ |
| `evaluate_robustness.py` | Multi-resolution robustness evaluation; flat-sequence dataset support |
| `utils.py` | DataLoader for `input1`/`input2` **and** flat sequential pairs |
| `output_inference.py` | Dynamic dataset length, auto mkdir, flat-sequence support |

## Stage 2 — reconstruction (`ImageReconstruction/`)

| File / feature | Description |
|----------------|-------------|
| `utils.py` | Max image size cap (VRAM), flexible 4-folder input layout |
| `inference.py` | Dynamic sample count, results paths |
| `constant.py` | Project-specific paths (`results2`, checkpoint dirs) |
| `vgg19/convert_vgg19.py` | VGG19 weight conversion for TF 1.x |

## Documentation & dataset

| Asset | Description |
|-------|-------------|
| `datasets/FERIT-Panorama26/` | Custom Mapillary dataset — **README + samples only on GitHub** |
| `docs/RESULTS.md` | Experiment metrics for CV / portfolio |
| `ImageReconstruction/UDIS-V1_REZULTATI_TRENIRANJA.md` | Stage 2 training log summary (local / thesis) |

## Named models (thesis / README)

| Name | Meaning |
|------|---------|
| **UDIS-Base** | Official pretrained Stage 1 (1M, H₃) |
| **UDIS-Net4-Joint** | Net4 + joint fine-tune (1.04M, H₄) |
| **UDIS-Net4-Frozen** | Net4-only fine-tune, frozen backbone (1.2M, H₄) |
| **UDIS-V1** | Custom Stage 2 reconstruction model |

## Environment

- `environment.yml`, `requirements.txt`, setup scripts for reproducibility on Windows.

**Skills to highlight:** Python, TensorFlow 1.x, OpenCV, CV pipelines, experiment design, PSNR/SSIM evaluation, technical writing.
