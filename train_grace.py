import argparse
import csv
import json
import logging
import os
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import monai
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets.dataloader import DatasetSegmentation, RandomGenerator, ValGenerator
from trainers.grace import build_grace
from utils.main_utils import load_cfg_from_cfg_file, read_text
from utils.reproducibility import (
    DEFAULT_SEED, capture_random_state, restore_random_state,
    seed_worker, set_random_seed,
)
from utils.pgc_grad_ops import (
    apply_bounded_meo,
    flatten_grads,
    pareto_coordinate,
    safe_cosine,
    unflatten_to_param_grads,
)


MODEL_STEM = "GRACE_UniMedCLIP_ViT-B-16"


def parse_args():
    parser = argparse.ArgumentParser(description="Train GRACE")
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("grace.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def resolve_num_workers(default: int = 8) -> int:
    value = os.getenv("GRACE_NUM_WORKERS")
    if value is not None:
        return max(0, int(value))
    return 0 if os.name == "nt" else default


def segmentation_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    bce_loss,
    dice_loss,
    cfg,
) -> torch.Tensor:
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    bce = bce_loss(logits, masks.float())
    dice_logits = logits.unsqueeze(1) if logits.ndim == 3 else logits
    dice_masks = masks.unsqueeze(1) if masks.ndim == 3 else masks
    dice = dice_loss(dice_logits, dice_masks.float())
    return cfg.TRAIN.CE_WEIGHT * bce + cfg.TRAIN.DICE_WEIGHT * dice


def consistency_loss(
    virtual_logits: torch.Tensor,
    detached_source_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:

    temperature = max(float(temperature), 1e-6)
    p_virt = torch.sigmoid(virtual_logits / temperature).clamp(1e-6, 1 - 1e-6)
    p_src = torch.sigmoid(detached_source_logits / temperature).clamp(
        1e-6, 1 - 1e-6
    )
    kl = p_src * torch.log(p_src / p_virt)
    kl += (1.0 - p_src) * torch.log((1.0 - p_src) / (1.0 - p_virt))
    return kl.mean()


def trainable_segmentation_parameters(model) -> Tuple[List[str], List[torch.nn.Parameter]]:
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and any(
            component in name
            for component in ("pvl_adapters", "mask_head", "upscale")
        )
    ]
    if not selected:
        selected = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    return [name for name, _ in selected], [parameter for _, parameter in selected]


def append_csv_row(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def atomic_torch_save(payload: Dict, path: Path) -> None:

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def tensor_stats(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item() if tensor.numel() > 1 else 0.0,
        "min": tensor.min().item(),
        "max": tensor.max().item(),
    }


def summarize_gcfp(aux_records: Sequence[Dict]) -> Dict[str, float]:
    if not aux_records:
        return {}
    weights = [
        record["mask_c"].detach().float().reshape(-1).cpu()
        for record in aux_records
        if torch.is_tensor(record.get("mask_c"))
    ]
    instability = [
        record["instability_score_c"].detach().float().reshape(-1).cpu()
        for record in aux_records
        if torch.is_tensor(record.get("instability_score_c"))
    ]
    variances = [
        record["var_c"].detach().float().reshape(-1).cpu()
        for record in aux_records
        if torch.is_tensor(record.get("var_c"))
    ]
    summary: Dict[str, float] = {}
    for name, chunks in (
        ("weight", weights),
        ("instability", instability),
        ("variance", variances),
    ):
        if chunks:
            stats = tensor_stats(torch.cat(chunks).tolist())
            summary.update({f"{name}_{key}": value for key, value in stats.items()})
    return summary


def validation_epoch(
    model,
    dataloader,
    device,
    bce_loss,
    dice_loss,
    cfg,
) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    dice_scores: List[float] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["ground_truth_mask"].to(device, non_blocking=True)
            logits = model(images, text=batch["text_prompt"], num_samples=1)[0]
            loss = segmentation_loss(logits, masks, bce_loss, dice_loss, cfg)
            losses.append(loss.item())

            predictions = (torch.sigmoid(logits) > 0.5).float()
            if predictions.ndim == 2:
                predictions = predictions.unsqueeze(0)
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)
            predictions = predictions.reshape(predictions.shape[0], -1)
            mask_flat = masks.float().reshape(masks.shape[0], -1)
            intersection = (predictions * mask_flat).sum(dim=1)
            denominator = predictions.sum(dim=1) + mask_flat.sum(dim=1)
            dice = (2.0 * intersection + 1e-7) / (denominator + 1e-7)
            dice_scores.extend(dice.detach().cpu().tolist())
    model.train()
    return mean(losses), mean(dice_scores)


def build_dataloaders(cfg):
    train_file = (
        f"Train_text_{cfg.data_percentage}.xlsx"
        if cfg.data_percentage != 100
        else "Train_text.xlsx"
    )
    val_file = (
        f"Val_text_{cfg.data_percentage}.xlsx"
        if cfg.data_percentage != 100
        else "Val_text.xlsx"
    )
    train_rows = read_text(os.path.join(cfg.DATASET.TEXT_PROMPT_PATH, train_file))
    val_rows = read_text(os.path.join(cfg.DATASET.TEXT_PROMPT_PATH, val_file))

    train_transform = transforms.Compose(
        [RandomGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])]
    )
    val_transform = ValGenerator(
        output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE]
    )
    train_dataset = DatasetSegmentation(
        cfg.DATASET.TRAIN_PATH,
        cfg.DATASET.NAME,
        train_rows,
        train_transform,
        image_size=cfg.DATASET.SIZE,
    )
    val_dataset = DatasetSegmentation(
        cfg.DATASET.VAL_PATH,
        cfg.DATASET.NAME,
        val_rows,
        val_transform,
        image_size=cfg.DATASET.SIZE,
    )

    common = {
        "batch_size": cfg.TRAIN.BATCH_SIZE,
        "worker_init_fn": seed_worker,
        "num_workers": resolve_num_workers(),
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(val_dataset, shuffle=False, **common),
    )


def overwrite_selected_gradients(
    flat_gradient: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> None:
    restored = unflatten_to_param_grads(flat_gradient, parameters)
    with torch.no_grad():
        for parameter, gradient in zip(parameters, restored):
            if parameter.grad is None:
                parameter.grad = gradient.detach().clone()
            else:
                parameter.grad.copy_(gradient.detach())


def main() -> None:
    cfg = parse_args()
    set_random_seed(cfg.seed)

    source_dataset_name = cfg.DATASET.NAME
    run_dataset_name = (
        source_dataset_name
        if cfg.data_percentage == 100
        else f"{source_dataset_name}_{cfg.data_percentage}"
    )
    cfg.DATASET.NAME = run_dataset_name

    run_dir = (
        Path(cfg.output_dir)
        / run_dataset_name
        / "trained_models"
        / f"seed{cfg.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(run_dir / "log_GRACE.txt")
    diagnostics_path = run_dir / "pgc_diagnostics.csv"
    if not cfg.resume and diagnostics_path.exists():
        diagnostics_path.unlink()

    logger.info("GRACE training configuration")
    logger.info(cfg)
    train_loader, val_loader = build_dataloaders(cfg)

    model = build_grace(cfg).to(cfg.MODEL.DEVICE)
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    pgc_parameter_names, pgc_parameters = trainable_segmentation_parameters(model)
    logger.info(f"Trainable parameters: {trainable_count}")
    logger.info(
        f"PGC coordinates {len(pgc_parameters)} tensors; "
        f"preview={pgc_parameter_names[:5]}"
    )

    use_gvos = bool(cfg.MODEL.USE_GVOS)
    use_gcfp = bool(cfg.MODEL.USE_GCFP)
    use_pgc = bool(cfg.MODEL.USE_PGC)
    lambda_virt = float(cfg.MODEL.LAMBDA_VIRT)
    lambda_clip = float(cfg.MODEL.LAMBDA_CLIP)
    lambda_cons = float(cfg.MODEL.LAMBDA_CONS)

    logger.info(
        "Method switches | "
        f"GVOS={use_gvos} GCFP={use_gcfp} PGC={use_pgc} "
        f"MEO={bool(cfg.MODEL.PGC_USE_MEO)}"
    )
    logger.info(
        "Method defaults | "
        f"K={cfg.MODEL.GVOS_STEPS} M={cfg.MODEL.GCFP_NUM_PROBES} "
        f"lambda_virt={lambda_virt} lambda_clip={lambda_clip} "
        f"lambda_cons={lambda_cons}"
    )

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=cfg.TRAIN.LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.TRAIN.NUM_EPOCHS,
        eta_min=cfg.TRAIN.MIN_LEARNING_RATE,
    )
    bce_loss = BCEWithLogitsLoss()
    dice_loss = monai.losses.DiceLoss(sigmoid=True, reduction="mean")

    latest_path = run_dir / f"{MODEL_STEM}_latest.pth"
    best_path = run_dir / f"{MODEL_STEM}_best_dice.pth"
    start_epoch = 0
    best_dice = 0.0
    if cfg.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint.get("best_dice", 0.0))
        if "random_state" in checkpoint:
            restore_random_state(checkpoint["random_state"])
        else:
            logger.warning("Checkpoint has no random state; exact continuation is unavailable.")
        logger.info(
            f"Resumed from epoch {start_epoch}; best Dice={best_dice:.4f}"
        )

    for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS):
        model.train()
        epoch_losses: Dict[str, List[float]] = {
            "total": [],
            "sup": [],
            "virt": [],
            "cons": [],
            "clip": [],
        }
        conflict_flags: List[float] = []
        cosines: List[float] = []
        meo_gammas: List[float] = []
        pgc_rules: Dict[str, int] = {}
        gcfp_records: List[Dict] = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images = batch["image"].to(cfg.MODEL.DEVICE, non_blocking=True)
            masks = batch["ground_truth_mask"].to(
                cfg.MODEL.DEVICE, non_blocking=True
            )
            texts = batch["text_prompt"]
            optimizer.zero_grad(set_to_none=True)

            z_src, text_features, height, width = model.extract_features(
                images, texts
            )
            z_virt = None
            if use_gvos:
                z_virt = model.generate_virtual_features(
                    z_src=z_src,
                    text_features=text_features,
                    mask=masks,
                    H=height,
                    W=width,
                    seg_loss_fn=lambda logits, target: segmentation_loss(
                        logits, target, bce_loss, dice_loss, cfg
                    ),
                    steps=int(cfg.MODEL.GVOS_STEPS),
                    step_size=float(cfg.MODEL.GVOS_STEP_SIZE),
                    eps=float(cfg.MODEL.GVOS_EPS),
                    patch_only=bool(cfg.MODEL.GVOS_PATCH_ONLY),
                    random_start=bool(cfg.MODEL.GVOS_RANDOM_START),
                    random_start_scale=float(cfg.MODEL.GVOS_RANDOM_START_SCALE),
                )

            if use_gcfp and z_virt is not None:
                z_src_used, z_virt_used, source_aux, _ = (
                    model.purify_source_and_virtual(
                        z_src=z_src,
                        z_virt=z_virt,
                        text_features=text_features,
                        H=height,
                        W=width,
                        num_probes=int(cfg.MODEL.GCFP_NUM_PROBES),
                        noise_std=float(cfg.MODEL.GCFP_NOISE_STD),
                        foreground_threshold=float(
                            cfg.MODEL.GCFP_FOREGROUND_THRESHOLD
                        ),
                        tau=float(cfg.MODEL.GCFP_TAU),
                        min_keep=float(cfg.MODEL.GCFP_MIN_KEEP),
                        max_keep=float(cfg.MODEL.GCFP_MAX_KEEP),
                        blend_beta=float(cfg.MODEL.GCFP_BLEND_BETA),
                        patch_only=bool(cfg.MODEL.GVOS_PATCH_ONLY),
                    )
                )
                gcfp_records.append(source_aux)
            elif use_gcfp:
                z_src_used, source_aux = model.purify_features(
                    z_src,
                    text_features,
                    height,
                    width,
                    num_probes=int(cfg.MODEL.GCFP_NUM_PROBES),
                    noise_std=float(cfg.MODEL.GCFP_NOISE_STD),
                    foreground_threshold=float(
                        cfg.MODEL.GCFP_FOREGROUND_THRESHOLD
                    ),
                    tau=float(cfg.MODEL.GCFP_TAU),
                    min_keep=float(cfg.MODEL.GCFP_MIN_KEEP),
                    max_keep=float(cfg.MODEL.GCFP_MAX_KEEP),
                    blend_beta=float(cfg.MODEL.GCFP_BLEND_BETA),
                    patch_only=bool(cfg.MODEL.GVOS_PATCH_ONLY),
                )
                z_virt_used = z_virt
                gcfp_records.append(source_aux)
            else:
                z_src_used = z_src
                z_virt_used = z_virt

            source_logits, _ = model.decode_from_features(
                z_src_used, text_features, height, width
            )
            loss_sup = segmentation_loss(
                source_logits, masks, bce_loss, dice_loss, cfg
            )
            loss_clip = model.compute_clip_loss_from_features(
                z_src, text_features
            )
            loss_virt = torch.zeros_like(loss_sup)
            loss_cons = torch.zeros_like(loss_sup)
            if z_virt_used is not None:
                virtual_logits, _ = model.decode_from_features(
                    z_virt_used,
                    text_features.detach(),
                    height,
                    width,
                )
                loss_virt = segmentation_loss(
                    virtual_logits, masks, bce_loss, dice_loss, cfg
                )
                loss_cons = consistency_loss(
                    virtual_logits,
                    source_logits.detach(),
                    temperature=float(cfg.MODEL.PGC_TEMPERATURE),
                )

            total_loss = loss_sup + lambda_clip * loss_clip
            if z_virt_used is not None:
                total_loss = total_loss + lambda_virt * loss_virt
                total_loss = total_loss + lambda_cons * loss_cons

            pgc_active = (
                use_pgc and z_virt_used is not None and bool(pgc_parameters)
            )
            if pgc_active:
                def flat_grad(objective: torch.Tensor) -> torch.Tensor:
                    gradients = torch.autograd.grad(
                        objective,
                        pgc_parameters,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )
                    return flatten_grads(gradients, pgc_parameters)

                g_sup = flat_grad(loss_sup)
                g_cons = flat_grad(lambda_cons * loss_cons)
                g_virt = flat_grad(lambda_virt * loss_virt)
                g_clip = flat_grad(lambda_clip * loss_clip)

                cosine = safe_cosine(
                    g_sup, g_cons, eps=float(cfg.MODEL.PGC_EPS)
                )
                g_pgc_bar, pgc_metadata = pareto_coordinate(
                    g_sup,
                    g_cons,
                    eps=float(cfg.MODEL.PGC_EPS),
                    small_norm_thresh=float(cfg.MODEL.PGC_SMALL_NORM_EPS),
                )
                g_pgc, meo_metadata = apply_bounded_meo(
                    g_pgc_bar,
                    g_sup,
                    g_cons,
                    cfg.MODEL,
                    conflict=bool(pgc_metadata["conflict"]),
                    eps=float(cfg.MODEL.PGC_EPS),
                )

                g_final = g_pgc + g_clip + g_virt
                if torch.isfinite(g_final).all():
                    total_loss.backward()
                    overwrite_selected_gradients(g_final, pgc_parameters)
                else:
                    logger.warning(
                        "Non-finite coordinated gradient; using standard sum."
                    )
                    total_loss.backward()

                conflict = bool(pgc_metadata["conflict"])
                rule = str(pgc_metadata["rule"])
                pgc_rules[rule] = pgc_rules.get(rule, 0) + 1
                conflict_flags.append(float(conflict))
                cosines.append(float(cosine.detach().item()))
                meo_gammas.append(float(meo_metadata["applied_gamma"]))
            else:
                total_loss.backward()

            optimizer.step()
            epoch_losses["total"].append(total_loss.item())
            epoch_losses["sup"].append(loss_sup.item())
            epoch_losses["virt"].append(loss_virt.item())
            epoch_losses["cons"].append(loss_cons.item())
            epoch_losses["clip"].append(loss_clip.item())

        scheduler.step()
        val_loss, val_dice = validation_epoch(
            model,
            val_loader,
            cfg.MODEL.DEVICE,
            bce_loss,
            dice_loss,
            cfg,
        )
        loss_means = {
            name: mean(values) if values else 0.0
            for name, values in epoch_losses.items()
        }
        cosine_summary = tensor_stats(cosines)
        gcfp_summary = summarize_gcfp(gcfp_records)
        conflict_ratio = mean(conflict_flags) if conflict_flags else 0.0
        mean_gamma = mean(meo_gammas) if meo_gammas else 1.0

        logger.info(
            f"EPOCH {epoch + 1}/{cfg.TRAIN.NUM_EPOCHS} | "
            f"train={loss_means['total']:.4f} val={val_loss:.4f} "
            f"Dice={val_dice:.4f}"
        )
        logger.info(
            "Losses | "
            f"L_sup={loss_means['sup']:.4f} "
            f"L_virt={loss_means['virt']:.4f} "
            f"L_cons={loss_means['cons']:.4f} "
            f"L_clip={loss_means['clip']:.4f}"
        )
        logger.info(
            "PGC | "
            f"conflict_ratio={conflict_ratio:.4f} "
            f"cos_mean={cosine_summary['mean']:.4f} "
            f"meo_gamma={mean_gamma:.4f} rules={pgc_rules}"
        )
        if gcfp_summary:
            logger.info(
                "GCFP | "
                f"weight_mean={gcfp_summary.get('weight_mean', float('nan')):.4f} "
                f"weight_min={gcfp_summary.get('weight_min', float('nan')):.4f} "
                f"instability_mean="
                f"{gcfp_summary.get('instability_mean', float('nan')):.4f}"
            )

        append_csv_row(
            diagnostics_path,
            {
                "epoch": epoch + 1,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": loss_means["total"],
                "val_loss": val_loss,
                "val_dice": val_dice,
                "loss_sup": loss_means["sup"],
                "loss_virt": loss_means["virt"],
                "loss_cons": loss_means["cons"],
                "loss_clip": loss_means["clip"],
                "pgc_conflict_ratio": conflict_ratio,
                "pgc_cosine_mean": cosine_summary["mean"],
                "meo_gamma_mean": mean_gamma,
                "pgc_rules": json.dumps(pgc_rules, sort_keys=True),
                "gcfp_weight_mean": gcfp_summary.get(
                    "weight_mean", float("nan")
                ),
                "gcfp_instability_mean": gcfp_summary.get(
                    "instability_mean", float("nan")
                ),
            },
        )

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice": max(best_dice, val_dice),
            "config": dict(cfg),
            "random_state": capture_random_state(),
        }
        if val_dice > best_dice:
            logger.info(f"New best Dice: {best_dice:.4f} -> {val_dice:.4f}")
            best_dice = val_dice
            checkpoint["best_dice"] = best_dice
            atomic_torch_save(checkpoint, best_path)
        atomic_torch_save(checkpoint, latest_path)

    logger.info(f"Training complete. Best Dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
