# Experiment summary (English)

Short reference for recruiters and reviewers. Full thesis text (Croatian) may live in other `docs/` files and is not required for understanding this repo.

---

## Model identifiers

| ID | Extractor | Matcher / extra | Fine-tuning data |
|----|-----------|-----------------|------------------|
| **LG-SIFT-MD** | SIFT (fixed) | LightGlue | MegaDepth |
| **LG-SIFT-DR-MD** | SIFT (fixed) | Descriptor refiner + LightGlue | MegaDepth |
| **LG-SIFT-CNN-MD** | SIFT keypoints + CNN descriptors | LightGlue (frozen) | MegaDepth |
| **LG-SIFT-FERIT** | SIFT (fixed) | LightGlue | FERIT-HOMOGRAPHY |
| **LG-SIFT-DR-FERIT** | SIFT (fixed) | Refiner + LightGlue | FERIT-HOMOGRAPHY |

---

## Homography mAA (public benchmarks)

| Model | HPatches | MegaDepth-1500 | ScanNet-1500 |
|-------|----------|----------------|--------------|
| LG-SIFT-MD | 0.456 | 0.605 | 0.274 |
| LG-SIFT-DR-MD | 0.461 | **0.622** | **0.293** |
| LG-SIFT-CNN-MD | 0.436 | 0.599 | 0.210 |
| LG-SIFT-FERIT | **0.468** | 0.581 | 0.259 |
| LG-SIFT-DR-FERIT | 0.453 | 0.587 | 0.252 |

**Takeaways**

- Descriptor refiner on MegaDepth (**LG-SIFT-DR-MD**) improves MegaDepth and ScanNet vs baseline.
- FERIT fine-tune (**LG-SIFT-FERIT**) peaks on HPatches but trades off on MegaDepth / ScanNet (domain shift).

---

## FERIT-HOMOGRAPHY (custom data)

- **978** train+val pairs, **465** scenes — see [`datasets/FERIT-HOMOGRAPHY/README.md`](datasets/FERIT-HOMOGRAPHY/README.md).
- Training validation (LG-SIFT-FERIT): ~**90%** match recall, val loss ~**0.20** (epoch 19).
- Held-out **test** set: homography mAA ~**4–5%** with RANSAC (much lower than HPatches ~0.47) — harder domain, full evaluation protocol in thesis notes.

Eval command:

```bash
python -m gluefactory.eval.ferit_homography \
  --conf <config> --checkpoint <run> --overwrite \
  model.extractor.backend=opencv data.num_workers=0 \
  data.root=<path> data.pairs=<path>/pairs_test.txt
```

---

## Descriptor refiner (LG-SIFT-DR-*)

Small MLP on 128-D SIFT descriptors: LayerNorm → Linear(128→256) → GELU → Linear(256→128), L2-normalized output. **~66k parameters**, final layer zero-init (identity at start). SIFT stays frozen; refiner + LightGlue train.

Implementation: `gluefactory/models/utils/descriptor_refiner.py`.

---

## Visualization tooling

`gluefactory/scripts/visualize_hpatches_matches.py`:

- Row 1: **SIFT + LightGlue**
- Row 2: **SuperPoint + LightGlue**
- Homography PNG: green/red matches vs GT **H**, green grid = GT warp, orange dashed = RANSAC **H**

Public demo uses **HPatches** (`v_graffiti`, etc.) — no FERIT images required on GitHub.

---

## Reproducibility notes

- Checkpoints: `outputs/training/<run_name>/` (local, not in git).
- Windows: `model.extractor.backend=opencv`, `data.num_workers=0`, `MPLBACKEND=Agg` for headless plots.
- Upstream base: [cvg/glue-factory](https://github.com/cvg/glue-factory), Apache-2.0.
