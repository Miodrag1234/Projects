"""Save side-by-side match visualizations for SIFT+LightGlue vs SuperPoint+LightGlue.

Outputs plain match figures and homography-aware figures (GT inlier coloring + corners).

Example:
    python -m gluefactory.scripts.visualize_hpatches_matches
    python -m gluefactory.scripts.visualize_hpatches_matches --sequence v_graffiti --query 5
    python -m gluefactory.scripts.visualize_hpatches_matches --homography_only --error_threshold 3
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from gluefactory.datasets.base_dataset import collate
from gluefactory.datasets.hpatches import HPatches
from gluefactory.eval.io import extract_benchmark_conf, load_model, parse_config_path
from gluefactory.geometry.homography import sym_homography_error, warp_points
from gluefactory.utils.tensor import batch_to_device, map_tensor, rbd
from gluefactory.visualization.viz2d import (
    add_text,
    plot_keypoints,
    plot_matches,
)

DEFAULT_MODELS = {
    "SIFT + LightGlue": "sift+lightglue-official",
    "SuperPoint + LightGlue": "superpoint+lightglue-official",
}


def load_pipeline(config_name, sift_backend=None):
    conf_path = parse_config_path(config_name, "configs/")
    conf = OmegaConf.load(conf_path)
    OmegaConf.resolve(conf)
    model_conf = extract_benchmark_conf(conf, "hpatches").model
    if sift_backend is not None and "extractor" in model_conf:
        model_conf.extractor.backend = sift_backend
    return load_model(model_conf, None)


def find_index(dataset, sequence, query):
    for i, (seq, q_idx, _) in enumerate(dataset.items):
        if seq == sequence and q_idx == query:
            return i
    raise ValueError(f"Pair not found: {sequence} image 1 -> {query}")


def tensors_to_cpu(batch):
    return map_tensor(
        batch,
        lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x,
    )


@torch.no_grad()
def run_match(model, data, device):
    data = batch_to_device(data, device)
    pred = model(data)
    pred = tensors_to_cpu(pred)
    data = tensors_to_cpu(data)
    return rbd(pred), rbd(data)


def view_image(view):
    img = view["image"]
    if isinstance(img, torch.Tensor) and img.ndim == 4:
        img = img[0]
    return img.permute(1, 2, 0).numpy()


def get_image_size(view):
    size = view["image_size"]
    if isinstance(size, torch.Tensor):
        size = size.cpu().numpy()
    if getattr(size, "ndim", 0) > 1:
        size = size[0]
    return size.astype(np.float32)


def get_matched_keypoints(pred):
    kp0 = pred["keypoints0"].numpy()
    kp1 = pred["keypoints1"].numpy()
    matches = pred["matches0"].numpy()
    valid = matches > -1
    return kp0, kp1, valid, kp0[valid], kp1[matches[valid]]


def image_corners(image_size):
    w, h = image_size[0], image_size[1]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def draw_quad(ax, corners, color, linestyle="-", lw=2.0, alpha=1.0):
    closed = np.concatenate([corners, corners[:1]], axis=0)
    ax.plot(
        closed[:, 0],
        closed[:, 1],
        color=color,
        linestyle=linestyle,
        lw=lw,
        alpha=alpha,
    )


def draw_warped_grid(ax, corners, n_cells=3, color="lime", alpha=0.35, lw=1.0):
    tl, tr, br, bl = corners
    for t in np.linspace(0, 1, n_cells + 1):
        left = tl * (1 - t) + bl * t
        right = tr * (1 - t) + br * t
        ax.plot(
            [left[0], right[0]],
            [left[1], right[1]],
            color=color,
            alpha=alpha,
            lw=lw,
        )
    for s in np.linspace(0, 1, n_cells + 1):
        top = tl * (1 - s) + tr * s
        bottom = bl * (1 - s) + br * s
        ax.plot(
            [top[0], bottom[0]],
            [top[1], bottom[1]],
            color=color,
            alpha=alpha,
            lw=lw,
        )


def estimate_homography_ransac(kpm0, kpm1, ransac_thresh=3.0):
    if len(kpm0) < 4:
        return None, None
    H, mask = cv2.findHomography(
        kpm0.astype(np.float64),
        kpm1.astype(np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    if H is None:
        return None, None
    return H.astype(np.float32), mask.ravel().astype(bool)


def homography_corner_error_np(H_est, H_gt, image_size):
    corners0 = image_corners(image_size)
    corners_gt = warp_points(corners0, H_gt, inverse=False)
    if H_est is None:
        return float("inf")
    corners_est = warp_points(corners0, H_est, inverse=False)
    return float(np.linalg.norm(corners_est - corners_gt, axis=1).mean())


def show_image_pair(imgs, axes_row):
    for ax, img in zip(axes_row, imgs):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_xlim([0, img.shape[1]])
        ax.set_ylim([img.shape[0], 0])


def make_model_axes(n_models, figsize_per_row=5):
    fig, axes = plt.subplots(n_models, 2, figsize=(14, figsize_per_row * n_models))
    if n_models == 1:
        axes = np.array([axes])
    return fig, axes


def draw_matches_row(data, pred, title, axes):
    img0 = view_image(data["view0"])
    img1 = view_image(data["view1"])
    kp0, kp1, valid, matched_kp0, matched_kp1 = get_matched_keypoints(pred)

    show_image_pair([img0, img1], axes)
    plot_keypoints([kp0, kp1], axes=axes, colors="royalblue", ps=3)
    if len(matched_kp0) > 0:
        plot_matches(matched_kp0, matched_kp1, axes=axes, a=0.65, lw=0.8, ps=0)

    add_text(0, title, axes=axes, fs=14)
    add_text(
        0,
        f"{len(kp0)} keypoints, {valid.sum()} matches",
        pos=(0.01, 0.03),
        fs=11,
        axes=axes,
    )


def draw_homography_row(data, pred, title, axes, error_threshold, ransac_thresh):
    img0 = view_image(data["view0"])
    img1 = view_image(data["view1"])
    kp0, kp1, valid, matched_kp0, matched_kp1 = get_matched_keypoints(pred)
    H_gt = np.asarray(data["H_0to1"], dtype=np.float32)
    if H_gt.ndim == 3:
        H_gt = H_gt[0]
    image_size = get_image_size(data["view0"])

    show_image_pair([img0, img1], axes)
    plot_keypoints([kp0, kp1], axes=axes, colors="royalblue", ps=3)

    n_inliers = 0
    corner_error = float("inf")
    H_est = None
    if len(matched_kp0) > 0:
        H_torch = torch.as_tensor(H_gt, dtype=torch.float32)
        k0 = torch.as_tensor(matched_kp0, dtype=torch.float32)
        k1 = torch.as_tensor(matched_kp1, dtype=torch.float32)
        errors = sym_homography_error(k0, k1, H_torch).numpy()
        inliers = errors < error_threshold
        n_inliers = int(inliers.sum())
        if n_inliers > 0:
            plot_matches(
                matched_kp0[inliers],
                matched_kp1[inliers],
                color="lime",
                axes=axes,
                a=0.75,
                lw=1.0,
                ps=0,
            )
        n_outliers = int((~inliers).sum())
        if n_outliers > 0:
            plot_matches(
                matched_kp0[~inliers],
                matched_kp1[~inliers],
                color="red",
                axes=axes,
                a=0.55,
                lw=0.8,
                ps=0,
            )

        H_est, _ = estimate_homography_ransac(
            matched_kp0, matched_kp1, ransac_thresh=ransac_thresh
        )
        corner_error = homography_corner_error_np(H_est, H_gt, image_size)

    corners0 = image_corners(image_size)
    corners1_gt = warp_points(corners0, H_gt, inverse=False)

    draw_quad(axes[0], corners0, color="cyan", linestyle="--", lw=1.8, alpha=0.9)
    draw_quad(axes[1], corners1_gt, color="lime", linestyle="-", lw=2.2, alpha=0.95)
    draw_warped_grid(axes[1], corners1_gt, n_cells=3, color="lime", alpha=0.25, lw=1.0)
    if H_est is not None:
        corners1_est = warp_points(corners0, H_est, inverse=False)
        draw_quad(
            axes[1], corners1_est, color="orange", linestyle="--", lw=2.0, alpha=0.95
        )

    add_text(0, title, axes=axes, fs=14)
    add_text(
        0,
        (
            f"{len(kp0)} kp, {valid.sum()} matches | "
            f"GT inliers (<{error_threshold:g}px): {n_inliers}/{valid.sum()} | "
            f"corner err: {corner_error:.2f}px"
        ),
        pos=(0.01, 0.03),
        fs=10,
        axes=axes,
    )
    add_text(
        1,
        "green=GT H, orange=est. H (RANSAC)",
        pos=(0.01, 0.03),
        fs=10,
        axes=axes,
    )


def save_pair_figure(fig, output_dir, filename, dpi, show):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_pair(
    dataset,
    models,
    device,
    idx,
    output_dir,
    dpi,
    show,
    save_matches,
    save_homography,
    error_threshold,
    ransac_thresh,
):
    data = collate([dataset[idx]])
    seq, q_idx, is_illu = dataset.items[idx]
    pair_name = f"{seq}_1_to_{q_idx}"
    tag = "illumination" if is_illu else "viewpoint"
    title_suffix = f"HPatches: {pair_name} ({tag})"

    if save_matches:
        fig, axes = make_model_axes(len(models))
        for row_axes, (label, model) in zip(axes, models.items()):
            pred, data_cpu = run_match(model, data, device)
            draw_matches_row(data_cpu, pred, label, row_axes)
        fig.suptitle(title_suffix, fontsize=16, y=0.995)
        fig.tight_layout()
        save_pair_figure(
            fig, output_dir, f"{pair_name}_{tag}_matches.png", dpi, show=False
        )

    if save_homography:
        fig, axes = make_model_axes(len(models))
        for row_axes, (label, model) in zip(axes, models.items()):
            pred, data_cpu = run_match(model, data, device)
            draw_homography_row(
                data_cpu,
                pred,
                label,
                row_axes,
                error_threshold=error_threshold,
                ransac_thresh=ransac_thresh,
            )
        fig.suptitle(
            f"{title_suffix} — homography (green=inlier, red=outlier)",
            fontsize=16,
            y=0.995,
        )
        fig.tight_layout()
        save_pair_figure(
            fig, output_dir, f"{pair_name}_{tag}_homography.png", dpi, show
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SIFT+LightGlue vs SuperPoint+LightGlue on HPatches."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/visualizations/hpatches"),
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Dataset indices to visualize (default: a few viewpoint pairs).",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="HPatches sequence name, e.g. v_graffiti",
    )
    parser.add_argument(
        "--query",
        type=int,
        default=5,
        help="Target image index in the sequence (2-6), used with --sequence",
    )
    parser.add_argument(
        "--sift_backend",
        type=str,
        default="opencv",
        help="SIFT backend (opencv is safest on Windows)",
    )
    parser.add_argument(
        "--error_threshold",
        type=float,
        default=3.0,
        help="GT homography inlier threshold in pixels (matches inspect viewer).",
    )
    parser.add_argument(
        "--ransac_thresh",
        type=float,
        default=3.0,
        help="RANSAC reprojection threshold for estimated homography.",
    )
    parser.add_argument(
        "--matches_only",
        action="store_true",
        help="Save only plain match visualizations.",
    )
    parser.add_argument(
        "--homography_only",
        action="store_true",
        help="Save only homography visualizations.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()

    save_matches = not args.homography_only
    save_homography = not args.matches_only
    if not save_matches and not save_homography:
        save_matches = save_homography = True

    data_conf = OmegaConf.merge(
        {
            "batch_size": 1,
            "num_workers": 0,
            "preprocessing": {
                "resize": 480,
                "side": "short",
            },
        },
        OmegaConf.from_cli(args.dotlist),
    )
    dataset = HPatches(data_conf)

    if args.sequence is not None:
        indices = [find_index(dataset, args.sequence, args.query)]
    elif args.indices is not None:
        indices = args.indices
    else:
        preferred = [
            ("v_graffiti", 5),
            ("v_bricks", 6),
            ("v_cube", 4),
        ]
        indices = []
        for seq, q in preferred:
            try:
                indices.append(find_index(dataset, seq, q))
            except ValueError:
                continue
        if not indices:
            indices = [0, 10, 20]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = {}
    for label, config_name in DEFAULT_MODELS.items():
        print(f"Loading {label} ({config_name})...")
        model = load_pipeline(config_name, sift_backend=args.sift_backend)
        models[label] = model.to(device).eval()

    for idx in indices:
        seq, q_idx, is_illu = dataset.items[idx]
        print(f"Visualizing [{idx}] {seq}: 1 -> {q_idx} ({'i' if is_illu else 'v'})")
        visualize_pair(
            dataset,
            models,
            device,
            idx,
            args.output_dir,
            args.dpi,
            args.show,
            save_matches=save_matches,
            save_homography=save_homography,
            error_threshold=args.error_threshold,
            ransac_thresh=args.ransac_thresh,
        )


if __name__ == "__main__":
    main()
