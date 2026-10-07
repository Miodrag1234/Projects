# Attribution & copyright

## Original work (UDIS)

Most code under `ImageAlignment/` and `ImageReconstruction/` is derived from the official implementation:

**Unsupervised Deep Image Stitching (UDIS)**  
Nie, Lang; Lin, Chunyu; Liao, Kang; Liu, Shuaicheng; Zhao, Yao  
*IEEE Transactions on Image Processing*, 2021  

- Repository: https://github.com/nie-lang/UnsupervisedDeepImageStitching  
- Paper: https://doi.org/10.1109/TIP.2021.3092828  

**Rules for this portfolio repo:**

1. Do **not** present the entire codebase as solely your original work — always credit UDIS authors.
2. Do **not** upload their pretrained checkpoints to GitHub; link to the official Google Drive / Baidu URLs in the README.
3. List **your** changes in [CONTRIBUTIONS.md](CONTRIBUTIONS.md).

## VGG-19 weights

Stage 2 uses VGG-19 from [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg).  
Do not commit `vgg19.npy`; generate it locally (`ImageReconstruction/vgg19/convert_vgg19.py`).

## FERIT-Panorama26 (Mapillary)

Mapillary images are subject to **per-image licenses** (often Creative Commons).  
This repo includes **documentation and sample pairs only**, not the full sequence.

## UDIS-D dataset

UDIS-D is provided by the UDIS authors under their terms — do not redistribute it via this repository.

## This repository

Treat as an **academic / research fork** with clear attribution. Your experiments (Net4, evaluation scripts, documentation) are described in [CONTRIBUTIONS.md](CONTRIBUTIONS.md).
