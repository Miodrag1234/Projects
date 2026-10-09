# FERIT-Panorama26 — sample frames

Five consecutive Mapillary frames from the local sequence (subset of the full dataset).  
Adjacent files form **evaluation pairs** for stitching, e.g. `(image_0104.jpg, image_0105.jpg)`.

| File | Role |
|------|------|
| `image_0103.jpg` | Frame *t* |
| `image_0104.jpg` | Frame *t+1* — pair with 0103 |
| `image_0105.jpg` | Frame *t+2* — pair with 0104 |
| `image_0106.jpg` | Frame *t+3* — pair with 0105 |
| `image_0107.jpg` | Frame *t+4* — pair with 0106 |

**Source:** [Mapillary](https://www.mapillary.com/) — research / portfolio samples only; full dataset not redistributed.

**Example pair (0104 → 0105):**

| Left (reference) | Right (warp target) |
|------------------|---------------------|
| ![image_0104](image_0104.jpg) | ![image_0105](image_0105.jpg) |
