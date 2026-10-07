import argparse
import glob
import os
import re
import sys

import numpy as np

try:
    import OpenEXR
    import Imath
except ImportError as exc:
    raise ImportError(
        "Missing OpenEXR/Imath. Install in this Python env with: "
        "python -m pip install OpenEXR Imath"
    ) from exc


def read_exr_depth(exr_path: str) -> np.ndarray:
    exr = OpenEXR.InputFile(exr_path)
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = list(header["channels"].keys())
    depth_channel = None
    for c in channels:
        cl = c.lower()
        if "depth" in cl or cl == "z":
            depth_channel = c
            break
    if depth_channel is None:
        # Single/BW EXR fallback
        depth_channel = "R" if "R" in channels else channels[0]

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    buf = exr.channel(depth_channel, pt)
    depth = np.frombuffer(buf, dtype=np.float32).reshape((height, width))
    return depth


def depth_to_disp(depth: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    disp = np.zeros_like(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1e-6) & (depth < 1e6)
    disp[valid] = (fx * baseline) / depth[valid]
    disp[~np.isfinite(disp)] = 0.0
    return disp


def save_pfm(path: str, image: np.ndarray, scale: float = 1.0) -> None:
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    image = np.flipud(image)
    with open(path, "wb") as f:
        f.write(b"Pf\n")
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode("ascii"))
        endian = image.dtype.byteorder
        if endian == "<" or (endian == "=" and sys.byteorder == "little"):
            scale = -scale
        f.write(f"{scale}\n".encode("ascii"))
        image.tofile(f)


def numeric_suffix(path: str):
    m = re.search(r"(\d+)(?=\.[^.]+$)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser(description="Convert depth EXR to disparity PFM.")
    parser.add_argument("--input_dir", required=True, help="Directory containing depth_*.exr")
    parser.add_argument("--pattern", default="depth_*.exr", help="Glob pattern for EXR files")
    parser.add_argument("--output_dir", required=True, help="Directory to save disp_XXX.pfm")
    parser.add_argument("--fx", type=float, required=True, help="Focal length in pixels")
    parser.add_argument("--baseline", type=float, required=True, help="Baseline in meters")
    parser.add_argument("--prefix", default="disp_", help="Output PFM filename prefix")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exr_files = glob.glob(os.path.join(args.input_dir, args.pattern))
    exr_files.sort(key=numeric_suffix)

    if not exr_files:
        raise FileNotFoundError(f"No EXR files matched: {os.path.join(args.input_dir, args.pattern)}")

    for idx, exr_path in enumerate(exr_files):
        depth = read_exr_depth(exr_path)
        disp = depth_to_disp(depth, fx=args.fx, baseline=args.baseline)
        out_name = f"{args.prefix}{idx:03d}.pfm"
        out_path = os.path.join(args.output_dir, out_name)
        save_pfm(out_path, disp)
        print(
            f"[{idx}] {os.path.basename(exr_path)} -> {out_name} | "
            f"depth min/max {float(np.nanmin(depth)):.6f}/{float(np.nanmax(depth)):.6f} | "
            f"disp min/max {float(np.nanmin(disp)):.6f}/{float(np.nanmax(disp)):.6f}"
        )


if __name__ == "__main__":
    main()
