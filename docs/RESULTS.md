# Results summary (portfolio)

Condensed metrics for the README / CV. Full analysis in the thesis.

## Stage 1 — UDIS-D test set (1105 pairs), scale = 1.0

| Model | PSNR ↑ | SSIM ↑ |
|-------|--------|--------|
| **UDIS-Base** | **22.56** | **0.695** |
| UDIS-Net4-Joint (H₄) | 21.38 | 0.660 |
| UDIS-Net4-Frozen (H₄) | 19.97 | 0.623 |

**Conclusion:** UDIS-Base remains best on the reference benchmark; Net4 modifications did not improve it.

## Stage 1 — robustness on UDIS-D (PSNR)

| Scale | UDIS-Base | Net4-Joint | Net4-Frozen |
|-------|-----------|------------|-------------|
| 1.00 | **22.56** | 21.38 | 19.97 |
| 0.75 | **23.55** | 22.50 | 21.26 |
| 0.50 | **23.63** | 22.66 | 21.66 |
| 0.25 | **22.67** | 21.80 | 21.25 |

Success rate: **100%** (1105/1105) for all models.

## FERIT-Panorama26 (202 pairs), scale = 1.0

| Model | PSNR ↑ | SSIM ↑ |
|-------|--------|--------|
| UDIS-Base | 14.08 | 0.323 |
| UDIS-Net4-Joint | 14.10 | 0.326 |
| UDIS-Net4-Frozen | 14.07 | 0.321 |

**Conclusion:** clear **domain shift** vs. UDIS-D; differences between models are negligible (< 0.15 dB).

## Stage 2 — UDIS-V1

- Reconstruction training to **100,000** iterations  
- Checkpoint stored locally (`model.ckpt-100000`), not in repo  
- Visual output: [`examples/stitched/`](../examples/stitched/)
