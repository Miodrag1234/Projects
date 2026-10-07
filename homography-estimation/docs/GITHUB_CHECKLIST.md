# GitHub publish checklist (English CV repo)

## Include

- [ ] Root `README.md` (English)
- [ ] `LICENSE` (Apache-2.0 from Glue Factory)
- [ ] `docs/examples/*.png` (HPatches viz)
- [ ] `docs/datasets/FERIT-HOMOGRAPHY/README.md` (format only — **no images**)
- [ ] `docs/EXPERIMENT_SUMMARY.md`
- [ ] Your code: `ferit_homography.py`, scripts, configs, refiner, fixes

## Exclude (`.gitignore` already helps)

- [ ] `data/` entire tree  
- [ ] `outputs/`  
- [ ] `venv/`, `venv_eval/`  
- [ ] Checkpoints, `.env`, secrets  

## GitHub setup

1. Fork [cvg/glue-factory](https://github.com/cvg/glue-factory) **or** new repo with clear “based on Glue Factory” in README.  
2. Keep upstream attribution and `LICENSE`.  
3. Pin the repo on your GitHub profile for CV.

## CV line (copy-paste)

> Extended [Glue Factory](https://github.com/cvg/glue-factory) for homography estimation: custom FERIT-HOMOGRAPHY dataset pipeline (docs + tooling), eval module, SIFT/LightGlue benchmarks on HPatches/MegaDepth/ScanNet, match & homography visualization — PyTorch, OpenCV.
