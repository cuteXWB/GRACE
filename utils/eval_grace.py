import argparse
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.SurfaceDice import (
    compute_dice_coefficient,
    compute_surface_dice_at_tolerance,
    compute_surface_distances,
)
from utils.main_utils import load_cfg_from_cfg_file


MODEL_STEM = "GRACE_UniMedCLIP_ViT-B-16"
VALID_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GRACE masks")
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--prompt-design", "--prompt_design", default="original")
    parser.add_argument("--data-percentage", "--data_percentage", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="output_grace")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Configuration overrides as KEY VALUE pairs",
    )
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update(vars(args))
    return cfg


def image_index(directory: Path):
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    }


def evaluate_pair(gt_path: Path, prediction_path: Path, tolerance: float = 2.0):
    gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    prediction = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise ValueError(f"Unreadable ground-truth mask: {gt_path}")
    if prediction is None:
        raise ValueError(f"Unreadable prediction mask: {prediction_path}")

    prediction = cv2.resize(
        prediction,
        (gt_mask.shape[1], gt_mask.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    gt_binary = gt_mask > 127
    prediction_binary = prediction > 127
    if not gt_binary.any() and not prediction_binary.any():
        return 1.0, 1.0
    if not gt_binary.any() or not prediction_binary.any():
        return 0.0, 0.0

    dsc = compute_dice_coefficient(gt_binary, prediction_binary)
    distances = compute_surface_distances(
        gt_binary[..., None],
        prediction_binary[..., None],
        [1, 1, 1],
    )
    nsd = compute_surface_dice_at_tolerance(distances, tolerance)
    return float(dsc), float(nsd)


def main() -> None:
    cfg = parse_args()
    dataset_name = cfg.DATASET.NAME
    output_dataset = (
        dataset_name
        if cfg.data_percentage == 100
        else f"{dataset_name}_{cfg.data_percentage}"
    )
    gt_dir = Path(cfg.DATASET.TEST_PATH) / "label"
    prediction_dir = (
        Path(cfg.output_dir)
        / output_dataset
        / "seg_results"
        / f"seed{cfg.seed}"
        / f"{MODEL_STEM}_Prompt-{cfg.prompt_design}"
    )
    csv_path = (
        Path(cfg.output_dir)
        / output_dataset
        / "seg_results"
        / f"seed{cfg.seed}"
        / f"test_GRACE_Prompt-{cfg.prompt_design}.csv"
    )
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")

    gt_files = image_index(gt_dir)
    prediction_files = image_index(prediction_dir)
    common_names = sorted(set(gt_files) & set(prediction_files))
    if not common_names:
        raise RuntimeError(
            f"No matching masks between {gt_dir} and {prediction_dir}"
        )
    print(f"Found {len(common_names)} matching files")

    metrics = OrderedDict(Name=[], DSC=[], NSD=[])
    with tqdm(common_names) as progress:
        for name in progress:
            dsc, nsd = evaluate_pair(
                gt_files[name], prediction_files[name], float(cfg.TEST.NSD_TOLERANCE)
            )
            metrics["Name"].append(gt_files[name].name)
            metrics["DSC"].append(round(dsc, 4))
            metrics["NSD"].append(round(nsd, 4))
            progress.set_postfix(
                {
                    "Mean DSC": f"{np.mean(metrics['DSC']):.4f}",
                    "Mean NSD": f"{np.mean(metrics['NSD']):.4f}",
                }
            )

    dataframe = pd.DataFrame(metrics)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(csv_path, index=False)
    average_dsc = dataframe["DSC"].mean()
    average_nsd = dataframe["NSD"].mean()
    print(">" * 20)
    print(
        f"Average DSC for {MODEL_STEM}_Prompt-{cfg.prompt_design} "
        f"{output_dataset}: {average_dsc * 100:.2f}%"
    )
    print(
        f"Average NSD for {MODEL_STEM}_Prompt-{cfg.prompt_design} "
        f"{output_dataset}: {average_nsd * 100:.2f}%"
    )
    print("<" * 20)


if __name__ == "__main__":
    main()
