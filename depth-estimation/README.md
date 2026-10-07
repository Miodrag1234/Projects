# Stereo Depth Estimation — Swappable Feature Backbones

[← Back to portfolio root](../README.md)

Stereo matching on **[Selective-IGEV](https://github.com/Windsrain/Selective-Stereo)** (CVPR 2024, MIT). The matching head is unchanged. This folder contains **my modifications**: swappable CNN backbones, training/eval scripts, and results on **KITTI 2015** and a custom Blender set (**Ferit-Depth26**).

Clone the original Selective-IGEV tree, then drop these files into it (or merge paths). Full datasets and `.pth` weights are **not** in this repo.

---

## What you can see without downloading data

| Asset | Location |
|-------|----------|
| **KITTI / Ferit-Depth26 tables** | [`rezultati_evaluacije/`](rezultati_evaluacije/) |
| **Dataset layout (no full set)** | [`examples/ferit-depth26/`](examples/ferit-depth26/) |
| **Backbone adapters** | [`core/extractor.py`](core/extractor.py) |
| **Train / eval** | [`train_stereo.py`](train_stereo.py), [`evaluate_stereo.py`](evaluate_stereo.py) |

---

## At a glance (for recruiters)

| | |
|---|---|
| **Problem** | Estimate stereo disparity (then depth) from a left–right image pair |
| **Stack** | PyTorch, timm, Selective-IGEV, KITTI, custom Blender data |
| **My work** | Swappable MobileNet / EfficientNet / ResNet / DenseNet extractors + adapters, partial Scene Flow checkpoint load, Ferit-Depth26 loader, EPE/D1 eval |
| **Evidence** | Tables below + CSVs in `rezultati_evaluacije/` |

---

## Results (lower is better)

### KITTI 2015

| Rank | Model | Backbone | EPE [px] | D1 [%] | Time [s] |
|---:|---|---|---:|---:|---:|
| 1 | **SI-MN3** | MobileNetV3 Large | **0.3030** | **1.4** | 0.239 |
| 2 | SI-R50 | ResNet50 | 0.3359 | 1.5 | 0.255 |
| 3 | SI-DN121 | DenseNet121 | 0.3409 | 1.6 | 0.259 |
| 4 | SI-EffB0 | EfficientNet-B0 | 0.3611 | 1.7 | 0.239 |
| 5 | SI-MN4 | MobileNetV4 Small | 0.3613 | 1.8 | 0.232 |
| 6 | SI-EffL0 | EfficientNet-Lite0 | 0.5126 | 1.9 | 0.237 |

### Ferit-Depth26 (custom synthetic test, 640×480)

| Rank | Model | Backbone | EPE [px] | D1 [%] |
|---:|---|---|---:|---:|
| 1 | **SI-R50** | ResNet50 | **2.2838** | 7.67 |
| 2 | SI-MN4 | MobileNetV4 Small | 2.4971 | 7.69 |
| 3 | SI-DN121 | DenseNet121 | 2.8281 | 7.26 |
| 4 | SI-EffL0 | EfficientNet-Lite0 | 3.7299 | 7.48 |
| 5 | SI-MN3 | MobileNetV3 Large | 3.8100 | **6.98** |
| 6 | SI-MN2 | MobileNetV2 (baseline) | 4.0142 | 13.08 |
| 7 | SI-EffB0 | EfficientNet-B0 | 5.5734 | 13.80 |

Best backbone **depends on the domain**: KITTI → SI-MN3; Ferit-Depth26 (EPE) → SI-R50. Most replacements beat the original MobileNetV2 baseline on Ferit-Depth26, especially D1 (~7% vs 13%).

---

## Scripts in this folder

| File | Role |
|------|------|
| `core/extractor.py` | Swappable backbones + 1×1 adapters |
| `core/stereo_datasets.py` | `SyntheticStereoTestDataset` (Ferit-Depth26 layout) |
| `train_stereo.py` | Fine-tune + partial checkpoint load |
| `evaluate_stereo.py` | KITTI and synthetic EPE/D1 |
| `exr_depth_to_pfm.py` | Blender EXR depth → disparity PFM (optional) |

`--feature_backbone` must match the checkpoint (`mobilenetv3`, `mobilenetv4`, `efficientnet_b0`, `efficientnet_lite0`, `resnet50`, `densenet121`, …).

---

## License

MIT — original Selective-Stereo copyright © 2024 Xianqi Wang. See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
