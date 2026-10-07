"""Generate Glue Factory pairs files from cropped_output dataset.

Each scene folder contains:
  - original.jpg (full scene image)
  - sceneXXX_patch_YY.png (crop)
  - sceneXXX_patch_YY_H_scene_to_crop.txt (homography: scene -> crop)

Output format for image_pairs with extra_data=homography:
  <image0> <image1> H11 H12 H13 H21 H22 H23 H31 H32 H33

Paths are relative to data/cropped_output (the dataset root).
"""

import argparse
import json
import random
from pathlib import Path

from gluefactory.settings import DATA_PATH


def read_homography(path: Path) -> list[str]:
    rows = []
    for line in path.read_text().strip().splitlines():
        rows.extend(line.split())
    if len(rows) != 9:
        raise ValueError(f"Expected 9 homography values in {path}, got {len(rows)}")
    return rows


def find_scene_image(scene_dir: Path) -> Path:
    for name in ("original.jpg", "original.png", "original.jpeg"):
        candidate = scene_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No original image found in {scene_dir}")


def collect_pairs(root: Path) -> list[tuple[str, str, list[str]]]:
    pairs = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        scene_image = find_scene_image(scene_dir)
        scene_rel = scene_image.relative_to(root).as_posix()

        for json_path in sorted(scene_dir.glob("*_patch_*.json")):
            meta = json.loads(json_path.read_text())
            crop_name = meta["crop_file"]
            h_name = meta["homography_scene_to_crop_file"]
            crop_path = scene_dir / crop_name
            h_path = scene_dir / h_name
            if not crop_path.exists():
                raise FileNotFoundError(crop_path)
            if not h_path.exists():
                raise FileNotFoundError(h_path)

            crop_rel = crop_path.relative_to(root).as_posix()
            h_elems = read_homography(h_path)
            pairs.append((scene_rel, crop_rel, h_elems))
    return pairs


def write_pairs(path: Path, pairs: list[tuple[str, str, list[str]]]) -> None:
    lines = [" ".join([img0, img1, *h]) for img0, img1, h in pairs]
    path.write_text("\n".join(lines) + "\n")


def split_by_scene(
    pairs: list[tuple[str, str, list[str]]],
    val_ratio: float,
    seed: int,
) -> tuple[list, list]:
    scenes = sorted({img0.split("/")[0] for img0, _, _ in pairs})
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n_val = max(1, int(len(scenes) * val_ratio))
    val_scenes = set(scenes[:n_val])

    train, val = [], []
    for pair in pairs:
        scene = pair[0].split("/")[0]
        (val if scene in val_scenes else train).append(pair)
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DATA_PATH / "cropped_output",
        help="Path to cropped_output dataset",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of scenes held out for validation",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Write all pairs to pairs_test.txt (no train/val split)",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    pairs = collect_pairs(root)
    if args.test_only:
        test_path = root / "pairs_test.txt"
        write_pairs(test_path, pairs)
        print(f"Total pairs: {len(pairs)}")
        print(f"Test pairs:  {len(pairs)} -> {test_path}")
        return

    train, val = split_by_scene(pairs, args.val_ratio, args.seed)

    train_path = root / "pairs_train.txt"
    val_path = root / "pairs_val.txt"
    write_pairs(train_path, train)
    write_pairs(val_path, val)

    print(f"Total pairs: {len(pairs)}")
    print(f"Train pairs: {len(train)} -> {train_path}")
    print(f"Val pairs:   {len(val)} -> {val_path}")
    print(f"Val scenes:  {len({p[0].split('/')[0] for p in val})}")


if __name__ == "__main__":
    main()
