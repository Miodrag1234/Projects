# Depth estimation

**Stereo / monocular depth estimation** (thesis chapter — *procjena dubine*).

[← Back to portfolio root](../README.md)

---

## Status

**In progress** — add implementation and evaluation here.

---

## Suggested layout

```
depth-estimation/
├── README.md
├── docs/
│   └── RESULTS.md          # metrics (e.g. abs rel, RMSE, δ thresholds)
├── examples/               # depth colormap previews (PNG)
├── src/ or Codes/
├── datasets/
│   └── README.md           # e.g. KITTI, SceneFlow — docs + samples only
└── requirements.txt
```

---

## Portfolio tips

- Add **2–3 depth visualizations** in `examples/` (colorized depth maps).
- Keep **one results table** in `docs/RESULTS.md` for CV readers.
- Do not commit large training sets or checkpoint files (use `.gitignore` like in `panorama-stitching/`).

---

## Possible link to panorama work

Depth maps can support stitching or overlap analysis in future work; keep cross-links in README only until code is shared.
