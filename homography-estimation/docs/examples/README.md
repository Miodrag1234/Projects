# Example figures for GitHub / CV

Commit **2–4 small PNG/JPG files** here so visitors see results without cloning datasets or checkpoints.

## Recommended files

| File in repo | How to create |
|--------------|----------------|
| `hpatches_matches.png` | `visualize_hpatches_matches` → `*_matches.png` |
| `hpatches_homography.png` | same script → `*_homography.png` |
| `ferit_pair.jpg` | (optional) anonymized scene+patch mosaic — only if you may publish it |
| `ferit_matches.png` | (optional) local viz on one FERIT pair |

## Generate HPatches examples

```bash
python -m gluefactory.scripts.visualize_hpatches_matches --sequence v_graffiti --query 5
```

Copy from `outputs/visualizations/hpatches/` into this folder. The root `README.md` embeds these images.

## Do not commit

- Full FERIT or MegaDepth image folders  
- Checkpoints (`.tar`)  
- Private / sensitive scenes without permission  

HPatches sequences such as `v_graffiti` are standard public benchmarks — safe for portfolio figures.
