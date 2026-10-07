# FERIT-HOMOGRAPHY (documentation only)

Custom homography dataset for **full scene ↔ local patch** matching.  
**Images are not included in this GitHub repository** (size and usage rights). This README describes the format so reviewers can understand the pipeline without access to the raw data.

---

## Task

Each sample is an image pair:

- **View 0** — full scene (`original.jpg`)
- **View 1** — cropped patch from the same scene
- **Ground truth** — planar homography **H** mapping coordinates from view 0 to view 1 (scene → crop)

Used for fine-tuning and evaluating **SIFT + LightGlue** with Glue Factory’s `image_pairs` dataset and `homography_matcher` supervision.

---

## Statistics (full collection)

| Split | Scenes | Pairs | List file |
|-------|:------:|:-----:|-----------|
| Train | 419 | 884 | `pairs_train.txt` |
| Val | 46 | 94 | `pairs_val.txt` |
| Test | separate hold-out | (project-specific) | `pairs_test.txt` |
| **Total (train+val)** | **465** | **978** | — |

Test scenes are kept in a **separate folder** from training; pair files are generated with the same script (`--test-only`).

---

## Folder layout (one scene)

```
scene042/
  original.jpg
  scene042_patch_03.png
  scene042_patch_03_H_scene_to_crop.txt
  scene042_patch_03.json          # metadata (patch index, file names, …)
  scene042_patch_04.png
  ...
```

- **`H_scene_to_crop.txt`** — 3×3 homography (row-major), maps **scene (view 0) → crop (view 1)**.
- **JSON** — references `crop_file`, `homography_scene_to_crop_file`, etc. (used by `generate_cropped_pairs.py`).

---

## Example pair line (`pairs_train.txt`)

Glue Factory `image_pairs` format with `extra_data: homography`:

```
<image0> <image1> H11 H12 H13 H21 H22 H23 H31 H32 H33
```

Concrete example (illustrative paths):

```
scene042/original.jpg scene042/scene042_patch_03.png 0.412 0.003 -128.5 -0.001 0.398 96.2 0.0000012 0.000004 1.0
```

See also: [`example_pair_line.txt`](example_pair_line.txt) in this folder.

---

## How pairs are generated

From the repository root (with your local copy of scenes under `data/cropped_output/`):

```bash
python -m gluefactory.scripts.generate_cropped_pairs --root data/cropped_output
python -m gluefactory.scripts.generate_cropped_pairs --root data/my_test_set --test-only
```

This writes `pairs_train.txt`, `pairs_val.txt`, and optionally `pairs_test.txt` under the dataset root.

---

## Training / evaluation in this project

| Step | Config / module |
|------|-----------------|
| Train on FERIT | `gluefactory/configs/sift+lightglue_cropped_homography.yaml` |
| Train + refiner | `gluefactory/configs/sift+lightglue_refiner_cropped_homography.yaml` |
| Eval (homography mAA) | `python -m gluefactory.eval.ferit_homography ...` |

Preprocessing (eval): resize long side 1024, `square_pad: true`, OpenCV homography estimator, `ransac_th: 0.5`.

---

## Example figures (optional in repo)

If you are allowed to publish **one anonymized scene**, add placeholders here:

| File | Description |
|------|-------------|
| `docs/examples/ferit_pair.jpg` | Mosaic or side-by-side scene + patch (no sensitive content) |
| `docs/examples/ferit_matches.png` | Matches overlay (generated locally) |

Otherwise, HPatches examples in [`docs/examples/`](../../examples/) demonstrate the same pipeline on public data.

---

## Citation / usage

Dataset collected for academic work at FERIT. Do not redistribute images without permission. Code in this repo for handling the format is under the same **Apache-2.0** license as Glue Factory.
