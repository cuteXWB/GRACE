import argparse
import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataloader import DatasetSegmentation, ValGenerator
from trainers.grace import build_grace
from utils.main_utils import load_cfg_from_cfg_file, normalize, read_text
from utils.reproducibility import DEFAULT_SEED, seed_worker, set_random_seed


MODEL_STEM = "GRACE_UniMedCLIP_ViT-B-16"


def parse_args():
    parser = argparse.ArgumentParser(description="Test GRACE")
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prompt-design", "--prompt_design", default="original")
    parser.add_argument("--data-percentage", "--data_percentage", type=int, default=100)
    parser.add_argument("--source-dataset", "--source_dataset", type=str)
    parser.add_argument("--output-dir", type=str, default="output_grace")
    parser.add_argument("--checkpoint", type=str, default="")
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


def configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("grace.test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def resolve_checkpoint(cfg, target_dataset_name: str) -> Path:
    if cfg.checkpoint:
        return Path(cfg.checkpoint)
    suffix = "latest" if bool(cfg.TEST.USE_LATEST) else "best_dice"
    if cfg.data_percentage != 100:
        trained_dataset = f"{cfg.source_dataset}_{cfg.data_percentage}"
    else:
        trained_dataset = cfg.source_dataset or target_dataset_name
    return (
        Path(cfg.output_dir)
        / trained_dataset
        / "trained_models"
        / f"seed{cfg.seed}"
        / f"{MODEL_STEM}_{suffix}.pth"
    )


def main() -> None:
    cfg = parse_args()
    set_random_seed(cfg.seed)
    target_dataset = cfg.DATASET.NAME
    cfg.source_dataset = cfg.source_dataset or target_dataset
    output_dataset = (
        target_dataset
        if cfg.data_percentage == 100
        else f"{target_dataset}_{cfg.data_percentage}"
    )
    cfg.DATASET.NAME = output_dataset

    results_root = (
        Path(cfg.output_dir)
        / output_dataset
        / "seg_results"
        / f"seed{cfg.seed}"
    )
    results_root.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(results_root / "log_GRACE.txt")
    checkpoint_path = resolve_checkpoint(cfg, target_dataset)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("GRACE inference configuration")
    logger.info(cfg)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(
        "Inference GCFP | "
        f"enabled={bool(cfg.MODEL.USE_GCFP) and bool(cfg.MODEL.GCFP_ON_EVAL)} "
        f"M={cfg.MODEL.GCFP_NUM_PROBES} "
        f"tau={cfg.MODEL.GCFP_TAU_EVAL} "
        f"w=[{cfg.MODEL.GCFP_MIN_KEEP_EVAL},"
        f"{cfg.MODEL.GCFP_MAX_KEEP_EVAL}] "
        f"beta={cfg.MODEL.GCFP_BLEND_BETA_EVAL}"
    )

    model = build_grace(cfg)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=cfg.MODEL.DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval().to(cfg.MODEL.DEVICE)

    prompt_file = f"Test_text_{cfg.prompt_design}.xlsx"
    rows = read_text(os.path.join(cfg.DATASET.TEXT_PROMPT_PATH, prompt_file))
    dataset = DatasetSegmentation(
        cfg.DATASET.TEST_PATH,
        cfg.DATASET.NAME,
        rows,
        ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE]),
        image_size=cfg.DATASET.SIZE,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.TEST.BATCH_SIZE),
        shuffle=False,
        worker_init_fn=seed_worker,
        num_workers=0 if os.name == "nt" else 8,
        pin_memory=torch.cuda.is_available(),
    )

    prediction_dir = (
        results_root / f"{MODEL_STEM}_Prompt-{cfg.prompt_design}"
    )
    uncertainty_dir = (
        Path(cfg.output_dir)
        / output_dataset
        / "uncertainty_results"
        / f"seed{cfg.seed}"
        / f"{MODEL_STEM}_Prompt-{cfg.prompt_design}"
    )
    prediction_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Test {output_dataset}"):
            samples = model(
                image=batch["image"].to(cfg.MODEL.DEVICE, non_blocking=True),
                text=batch["text_prompt"],
                num_samples=int(cfg.TEST.NUM_SAMPLES),
            )
            probabilities = torch.sigmoid(samples)
            mean_probability = probabilities.mean(dim=0)
            uncertainty = -(
                mean_probability * torch.log(mean_probability + 1e-8)
                + (1.0 - mean_probability)
                * torch.log(1.0 - mean_probability + 1e-8)
            )
            masks = mean_probability > float(cfg.TEST.MASK_THRESHOLD)

            for index, mask_name in enumerate(batch["mask_name"]):
                prediction = masks[index].cpu().numpy().astype(np.uint8) * 255
                cv2.imwrite(str(prediction_dir / mask_name), prediction)

                uncertainty_map = normalize(uncertainty[index].cpu().numpy())
                color = (
                    plt.get_cmap("nipy_spectral")(uncertainty_map)[:, :, :3]
                    * 255
                ).astype(np.uint8)
                cv2.imwrite(
                    str(uncertainty_dir / mask_name),
                    cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
                )

    logger.info(f"Predictions: {prediction_dir}")


if __name__ == "__main__":
    main()
