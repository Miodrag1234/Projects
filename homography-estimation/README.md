# Sparse Image Matching & Homography Estimation (SIFT / SuperPoint + LightGlue)

Extension of **[Glue Factory](https://github.com/cvg/glue-factory)** (Apache-2.0) for **homography estimation**: custom **FERIT-HOMOGRAPHY** dataset tooling, evaluation, training configs, and visualization. Public benchmarks: **HPatches**, **MegaDepth-1500**, **ScanNet-1500**.

This repository is a **fork with my additions** (not a from-scratch framework). Upstream code remains attributed to CVG / Glue Factory authors.

---

## What you can see without downloading data

| Asset | Location |
|-------|----------|
| **Match & homography figures** | [`docs/examples/`](docs/examples/) (PNG) |
| **Dataset format (no images shipped)** | [`docs/datasets/FERIT-HOMOGRAPHY/README.md`](docs/datasets/FERIT-HOMOGRAPHY/README.md) |
| **Experiment summary (English)** | [`docs/EXPERIMENT_SUMMARY.md`](docs/EXPERIMENT_SUMMARY.md) |
| **Runnable scripts** | `gluefactory/scripts/`, `gluefactory/eval/ferit_homography.py` |
| **Training / eval configs** | `gluefactory/configs/sift+lightglue_*cropped*.yaml` |

![Example: SIFT vs SuperPoint matches](docs/examples/hpatches_matches.png)
*Add this file locally — see [`docs/examples/README.md`](docs/examples/README.md).*

![Example: homography inliers & GT warp](docs/examples/hpatches_homography.png)

---

## At a glance (for recruiters)

| | |
|---|---|
| **Problem** | Estimate a planar homography from sparse feature matches between two views |
| **Stack** | PyTorch, Glue Factory, LightGlue, OpenCV SIFT, RANSAC |
| **My work** | FERIT dataset pipeline, homography eval module, descriptor refiner integration, HPatches visualization tool, Windows training fixes |
| **Evidence** | Benchmark tables below + scripts + example figures in `docs/examples/` |

---

## Benchmark results (homography mAA)

| Model ID | HPatches | MegaDepth-1500 | ScanNet-1500 |
|----------|----------|----------------|--------------|
| LG-SIFT-MD | 0.456 | 0.605 | 0.274 |
| **LG-SIFT-DR-MD** | 0.461 | **0.622** | **0.293** |
| LG-SIFT-CNN-MD | 0.436 | 0.599 | 0.210 |
| **LG-SIFT-FERIT** | **0.468** | 0.581 | 0.259 |
| LG-SIFT-DR-FERIT | 0.453 | 0.587 | 0.252 |

Model IDs and training setup: [`docs/EXPERIMENT_SUMMARY.md`](docs/EXPERIMENT_SUMMARY.md). Checkpoints are **not** hosted in this repo (size); metrics are reproducible via eval commands in the experiment doc.

---

## My contributions (high level)

- **`gluefactory/eval/ferit_homography.py`** — HPatches-style homography metrics on custom `image_pairs` data
- **`gluefactory/scripts/generate_cropped_pairs.py`** — build `pairs_train/val/test.txt` with GT homographies
- **`gluefactory/scripts/visualize_hpatches_matches.py`** — export PNGs: SIFT vs SuperPoint, GT homography coloring
- **`gluefactory/models/utils/descriptor_refiner.py`** — lightweight descriptor refinement before LightGlue (~66k params, identity init)
- **Configs** — `sift+lightglue_cropped_homography.yaml`, `sift+lightglue_refiner_cropped_homography.yaml`
- **Stability** — SIFT / homography dtype fixes, Windows-friendly training (`opencv` backend, `num_workers=0`)

---

## Quick start

Install (same as upstream):

```bash
conda create -n glue_factory python=3.10
conda activate glue_factory
pip install -e .
```

Full upstream instructions: [Glue Factory README](https://github.com/cvg/glue-factory/blob/main/README.md).

**Windows:** use `model.extractor.backend=opencv` and `data.num_workers=0` for SIFT training/eval.

### Visual demo (HPatches — download dataset locally or use existing `data/hpatches-sequences-release/`)

```bash
python -m gluefactory.scripts.visualize_hpatches_matches --sequence v_graffiti --query 5
```

Outputs go to `outputs/visualizations/hpatches/` (gitignored). Copy PNGs to `docs/examples/` for GitHub.

### FERIT-HOMOGRAPHY (you provide data — see dataset README)

```bash
python -m gluefactory.scripts.generate_cropped_pairs --root data/cropped_output

python -m gluefactory.eval.ferit_homography \
  --conf gluefactory/configs/sift+lightglue_cropped_homography.yaml \
  --checkpoint sift+lg_cropped_baseline --tag LG-SIFT-FERIT --overwrite \
  model.extractor.backend=opencv data.num_workers=0 \
  data.root=data/cropped_output data.pairs=data/cropped_output/pairs_test.txt
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_cropped_pairs.py` | Create pair list files from scene + patch folders |
| `visualize_hpatches_matches.py` | Side-by-side match & homography figures |
| `eval/ferit_homography.py` | mAA, RANSAC/DLT homography error on FERIT-style pairs |
| `eval/hpatches.py` | Standard HPatches eval (upstream + your checkpoints) |

Interactive inspection after eval (upstream):  
`python -m gluefactory.eval.inspect hpatches <tag1> <tag2> --default_plot homography`

---

## What is **not** in this GitHub repo

| Item | Why |
|------|-----|
| `data/` (FERIT images, MegaDepth, …) | Size & licensing — **dataset described in docs only** |
| Training checkpoints | Multi-GB; mention run names in CV |
| Full thesis PDF | University copyright — **English summary in `docs/EXPERIMENT_SUMMARY.md`** |

---

## License & attribution

- **Glue Factory** — [Apache License 2.0](LICENSE); see [cvg/glue-factory](https://github.com/cvg/glue-factory).
- **LightGlue / SuperPoint** — follow upstream licenses.
- **My additions** (scripts, eval, dataset docs, configs listed above) — Apache-2.0, same as the base project, with upstream copyright notices retained.

---

## Contact

*[Your Name]* · *[email]* · *[LinkedIn]*

**CV one-liner:** *Extended Glue Factory for homography estimation: FERIT-HOMOGRAPHY pipeline, custom eval, SIFT+LightGlue benchmarks (HPatches / MegaDepth / ScanNet), visualization tooling — PyTorch, OpenCV.*
