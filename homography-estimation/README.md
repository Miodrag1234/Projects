# Homography estimation

Standalone **homography / image alignment** work (thesis chapter — *procjena homografije*).

[← Back to portfolio root](../README.md)

---

## Status

**In progress** — add code, notebooks, and evaluation here when ready.

---

## Suggested layout (when you add files)

```
homography-estimation/
├── README.md
├── docs/
│   └── RESULTS.md          # PSNR, success rate, tables
├── examples/               # 2–3 visual pairs (input / warped)
├── src/ or Codes/            # scripts / models
├── datasets/
│   └── README.md             # dataset description only (no bulk images)
└── requirements.txt          # or use conda env at repo root
```

---

## Relation to panorama project

Stage 1 in [`panorama-stitching/`](../panorama-stitching/) already implements **deep homography** (UDIS).  
This folder can hold:

- classical baseline (SIFT + RANSAC),
- separate deep homography experiments,
- or exported modules reused from panorama Stage 1.

Avoid duplicating large UDIS checkpoints — link to download instructions instead.

---

## What to put on GitHub

| Include | Skip |
|---------|------|
| Source code, README, `docs/RESULTS.md` | Full UDIS-D / FERIT image folders |
| 1–2 example warps in `examples/` | `.ckpt` / large weights |
| Short metrics table | Private thesis PDF (unless you choose to share) |
