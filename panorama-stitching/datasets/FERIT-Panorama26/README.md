# FERIT-Panorama26

Custom evaluation dataset for **street-level panoramic stitching**, used in this project alongside the public **UDIS-D** benchmark.

> **This folder does not contain the full dataset.** Only documentation and a **few sample image pairs** are stored in the repository (see [`samples/`](samples/)). Full sequences remain local due to size and Mapillary licensing.

---

## Purpose

- Test model generalization **outside UDIS-D** (domain shift: Mapillary urban scenes).
- Support sequential pairing: adjacent frames along a driving path form image pairs for homography estimation.

---

## Source

- Images collected from **[Mapillary](https://www.mapillary.com/)** (crowdsourced street-level imagery with geolocation).
- Selection criteria: continuous routes, sufficient overlap between consecutive frames, urban / road scenes.

---

## Format (local project layout)

When prepared for training/evaluation locally, images are stored as a **flat sequence**:

```
image_0082.jpg
image_0083.jpg
...
image_1080.jpg
```

The data loader pairs consecutive images: `(image_i, image_{i+1})` after sorting by filename.

**Local folder name in code:** `ImageAlignment/testing2/` (not uploaded to GitHub).

---

## Statistics (full local copy)

| Property | Value |
|----------|--------|
| Images | 999 (local) |
| Pairs (sequential) | 499 |
| Format | JPEG |
| Naming | `image_XXXX.jpg` |
| Eval subset used in experiments | 202 pairs (machine-specific split) |

---

## Sample files in this repo

Add 1–2 representative pairs under [`samples/`](samples/):

```
samples/
├── pair_01_left.jpg
├── pair_01_right.jpg
└── README.md
```

*(Copy from your local `testing2/` — do not commit the full folder.)*

---

## Licensing & ethics

- Mapillary images are user-contributed under **per-image licenses** (often Creative Commons variants).
- Do **not** redistribute the full dataset via GitHub without checking each image’s license.
- In academic work: cite Mapillary as the source and state that data was used for research evaluation only.

---

## Related results

Evaluation on FERIT-Panorama26 (202 pairs, scale = 1.0):

| Model | PSNR | SSIM |
|-------|------|------|
| UDIS-Base | 14.08 | 0.323 |
| UDIS-Net4-Joint | 14.10 | 0.326 |
| UDIS-Net4-Frozen | 14.07 | 0.321 |

See [docs/RESULTS.md](../../docs/RESULTS.md) for robustness across resolutions.
