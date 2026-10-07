"""Evaluate homography estimation on a custom image_pairs dataset (e.g. FERIT-HOMOGRAPHY).

Uses the same homography metrics as HPatches (H_error_ransac_mAA, @3px, @5px, ...).
"""

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import EVAL_PATH
from ..utils.export_predictions import export_predictions
from ..utils.tensor import map_tensor
from ..utils.tools import AUCMetric
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import (
    eval_homography_dlt,
    eval_homography_robust,
    eval_matches_homography,
    eval_poses,
)


class FeritHomographyPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "image_pairs",
            "root": "cropped_output",
            "pairs": "cropped_output/pairs_test.txt",
            "extra_data": "homography",
            "batch_size": 1,
            "num_workers": 0,
            "preprocessing": {
                "resize": 1024,
                "side": "long",
                "square_pad": True,
            },
        },
        "model": {
            "ground_truth": {
                "name": None,
            }
        },
        "eval": {
            "estimator": "opencv",
            "ransac_th": 0.5,
        },
    }

    export_keys = [
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    ]
    optional_export_keys = []

    def _init(self, conf):
        pass

    def get_dataloader(self, data_conf=None):
        if data_conf is None:
            data_conf = self.conf.data
        dataset = get_dataset(data_conf.name)(data_conf)
        return dataset.get_data_loader("val")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run_eval(self, loader, pred_file):
        assert pred_file.exists()
        results = defaultdict(list)
        conf = self.conf.eval

        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )
        pose_results = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()

        for data in tqdm(loader):
            pred = cache_loader(data)
            data = map_tensor(data, lambda t: torch.squeeze(t, dim=0))

            if "keypoints0" in pred:
                results_i = eval_matches_homography(data, pred)
                results_i = {**results_i, **eval_homography_dlt(data, pred)}
            else:
                results_i = {}

            for th in test_thresholds:
                pose_results_i = eval_homography_robust(
                    data,
                    pred,
                    {"estimator": conf.estimator, "ransac_th": th},
                )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

            name = data["name"]
            if isinstance(name, (list, tuple)):
                name = name[0]
            results_i["names"] = name
            results_i["scenes"] = name.split("/")[0] if "/" in name else name

            for k, v in results_i.items():
                results[k].append(v)

        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(arr.dtype, np.number):
                continue
            summaries[f"m{k}"] = round(float(np.median(arr)), 3)

        auc_ths = [1, 3, 5]
        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=auc_ths, key="H_error_ransac", unit="px"
        )
        if "H_error_dlt" in results:
            dlt_aucs = AUCMetric(auc_ths, results["H_error_dlt"]).compute()
            for i, ath in enumerate(auc_ths):
                summaries[f"H_error_dlt@{ath}px"] = dlt_aucs[i]

        results = {**results, **pose_results[best_th]}
        summaries = {**summaries, **best_pose_results}

        figures = {}
        if "H_error_dlt" in results and "H_error_ransac" in results:
            figures["homography_recall"] = plot_cumulative(
                {
                    "DLT": results["H_error_dlt"],
                    self.conf.eval.estimator: results["H_error_ransac"],
                },
                [0, 10],
                unit="px",
                title="Homography",
            )

        return summaries, figures, results


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(FeritHomographyPipeline.default_conf)
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = FeritHomographyPipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir, overwrite=args.overwrite, overwrite_eval=args.overwrite_eval
    )

    pprint(s)
    if args.plot:
        for fig_name, fig in f.items():
            fig.canvas.manager.set_window_title(fig_name)
        plt.show()
