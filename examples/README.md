# Portfolio examples

Small, recruiter-friendly assets only (**no bulk exports** from `results/` or datasets).

## Current files

| Path | Description |
|------|-------------|
| [`stitched/panorama_000030.jpg`](stitched/panorama_000030.jpg) | Final panorama (Stage 2, **UDIS-V1**) |

## Recommended additions (optional)

```
examples/
├── input/
│   ├── pair_01_a.jpg      # source frame 1
│   └── pair_01_b.jpg      # source frame 2
├── aligned/               # optional Stage 1 warp preview
└── stitched/
    └── panorama_*.jpg     # 2–3 best outputs total
```

Keep each image **under ~500 KB** when possible.

## Copy from local results (PowerShell)

```powershell
Copy-Item ImageReconstruction\results\000030.jpg examples\stitched\
Copy-Item path\to\left.jpg  examples\input\pair_01_a.jpg
Copy-Item path\to\right.jpg examples\input\pair_01_b.jpg
```

These paths are referenced from the root [README.md](../README.md).
