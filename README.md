# Projects — Computer Vision Portfolio

**GitHub:** [Miodrag1234/Projects](https://github.com/Miodrag1234/Projects)

One repository for MSc / CV work: **panorama stitching**, **homography estimation**, and **depth estimation**. Each topic lives in its own folder with code, docs, and sample outputs (no large datasets or weights on GitHub).

---

## Projects

| Folder | Topic | Status | Quick link |
|--------|--------|--------|------------|
| [**panorama-stitching/**](panorama-stitching/) | Deep panoramic stitching (UDIS, Net4, FERIT-Panorama26) | **Done** — code, metrics, sample panorama | [README](panorama-stitching/README.md) · [results](panorama-stitching/docs/RESULTS.md) |
| [**homography-estimation/**](homography-estimation/) | SIFT / SuperPoint + LightGlue, FERIT-HOMOGRAPHY, benchmarks | **Done** — scripts, eval, docs | [README](homography-estimation/README.md) · [experiments](homography-estimation/docs/EXPERIMENT_SUMMARY.md) |
| [**depth-estimation/**](depth-estimation/) | Stereo / monocular depth | **Planned / in progress** | [README](depth-estimation/README.md) |

---

## Highlights for recruiters

- **Panorama:** end-to-end TF 1.x pipeline, custom Net4 architecture experiments, PSNR/SSIM + multi-resolution robustness, custom Mapillary dataset (documented).
- **Homography:** Glue Factory extensions — custom dataset tooling, homography eval, descriptor refiner, HPatches / MegaDepth / ScanNet mAA tables, match & homography visualizations.
- **Depth:** add your code here as you finish thesis chapters — same repo, clear separation.

---

## Sample output (panorama)

![Panorama example](panorama-stitching/examples/stitched/panorama_000030.jpg)

---

## Repository layout

```
Projects/                          ← this repo (root)
├── README.md                      ← you are here
├── panorama-stitching/
├── homography-estimation/         ← Glue Factory extensions (this project)
└── depth-estimation/
```

---

## Clone & update

```bash
git clone https://github.com/Miodrag1234/Projects.git
cd Projects
```

After changes:

```bash
git add -A
git commit -m "Update homography-estimation"
git push
```

Guide: [panorama-stitching/docs/PUSH_TO_GITHUB.md](panorama-stitching/docs/PUSH_TO_GITHUB.md)

---

## Tech stack (overall)

Python · TensorFlow 1.x / PyTorch (per project) · OpenCV · NumPy · Computer Vision · Deep Learning

---

## Contact

**Miodrag** · *[your email]* · *[LinkedIn]*
