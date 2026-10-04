from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def flatten_grads(
    grads: Iterable[torch.Tensor | None],
    params: Sequence[torch.nn.Parameter],
) -> torch.Tensor:

    grad_list = list(grads)
    param_list = list(params)
    if len(grad_list) != len(param_list):
        raise ValueError(
            f"Gradient/parameter length mismatch: {len(grad_list)} vs "
            f"{len(param_list)}"
        )
    chunks = [
        (torch.zeros_like(param) if grad is None else grad).reshape(-1)
        for grad, param in zip(grad_list, param_list)
    ]
    if not chunks:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def unflatten_to_param_grads(
    flat_grad: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
) -> List[torch.Tensor]:

    if flat_grad.ndim != 1:
        raise ValueError(f"flat_grad must be 1D, got {tuple(flat_grad.shape)}")

    outputs: List[torch.Tensor] = []
    offset = 0
    for param in params:
        next_offset = offset + param.numel()
        piece = flat_grad[offset:next_offset]
        if piece.numel() != param.numel():
            raise ValueError("flat_grad is shorter than the parameter vector")
        outputs.append(piece.view_as(param))
        offset = next_offset
    if offset != flat_grad.numel():
        raise ValueError("flat_grad contains trailing elements")
    return outputs


def safe_cosine(
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:

    if grad_a.shape != grad_b.shape:
        raise ValueError(
            f"Gradient shape mismatch: {tuple(grad_a.shape)} vs "
            f"{tuple(grad_b.shape)}"
        )
    if not torch.isfinite(grad_a).all() or not torch.isfinite(grad_b).all():
        return torch.zeros((), device=grad_a.device, dtype=grad_a.dtype)

    norm_product = grad_a.norm(p=2) * grad_b.norm(p=2)
    if norm_product.item() <= float(eps):
        return torch.zeros((), device=grad_a.device, dtype=grad_a.dtype)
    cosine = torch.dot(grad_a, grad_b) / (norm_product + float(eps))
    return cosine.clamp(-1.0, 1.0)


def pareto_coordinate(
    g_sup: torch.Tensor,
    g_cons: torch.Tensor,
    eps: float = 1e-12,
    small_norm_thresh: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    if g_sup.shape != g_cons.shape:
        raise ValueError(
            f"Gradient shape mismatch: {tuple(g_sup.shape)} vs "
            f"{tuple(g_cons.shape)}"
        )

    if not torch.isfinite(g_sup).all() or not torch.isfinite(g_cons).all():
        fallback = torch.nan_to_num(g_sup) + torch.nan_to_num(g_cons)
        return fallback, {
            "rule": "fallback_nonfinite_sum",
            "conflict": False,
            "dot": float("nan"),
            "projection_coefficient": 0.0,
        }

    norm_sup = g_sup.norm(p=2)
    norm_cons = g_cons.norm(p=2)
    dot = torch.dot(g_sup, g_cons)
    metadata: Dict[str, Any] = {
        "rule": "compatible_sum",
        "conflict": False,
        "dot": float(dot.detach().item()),
        "norm_sup": float(norm_sup.detach().item()),
        "norm_cons": float(norm_cons.detach().item()),
        "projection_coefficient": 0.0,
    }

    if (
        norm_sup.item() <= float(small_norm_thresh)
        or norm_cons.item() <= float(small_norm_thresh)
    ):
        metadata["rule"] = "degenerate_sum"
        return g_sup + g_cons, metadata

    if dot.item() >= 0.0:
        return g_sup + g_cons, metadata

    mu = dot / (torch.dot(g_sup, g_sup) + float(eps))
    g_pgc_bar = g_sup + g_cons - mu * g_sup
    metadata.update(
        {
            "rule": "conflict_projection",
            "conflict": True,
            "projection_coefficient": float(mu.detach().item()),
        }
    )
    return g_pgc_bar, metadata


def apply_bounded_meo(
    g_pgc_bar: torch.Tensor,
    g_sup: torch.Tensor,
    g_cons: torch.Tensor,
    cfg: Any,
    conflict: bool | None = None,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    use_meo = bool(_cfg_get(cfg, "PGC_USE_MEO", True))
    gamma_min = float(_cfg_get(cfg, "PGC_MEO_GAMMA_MIN", 1.0))
    gamma_max = float(_cfg_get(cfg, "PGC_MEO_GAMMA_MAX", 1.10))
    if gamma_min <= 0.0 or gamma_max < gamma_min:
        raise ValueError(
            f"Invalid MEO gamma range: [{gamma_min}, {gamma_max}]"
        )

    if conflict is None:
        conflict = bool(torch.dot(g_sup, g_cons).item() < 0.0)

    metadata: Dict[str, Any] = {
        "enabled": use_meo,
        "conflict": conflict,
        "raw_gamma": 1.0,
        "applied_gamma": 1.0,
        "rule": "identity",
    }
    if not use_meo:
        metadata["rule"] = "disabled"
        return g_pgc_bar, metadata
    if not conflict:
        metadata["rule"] = "compatible_identity"
        return g_pgc_bar, metadata

    coordinated_norm = g_pgc_bar.norm(p=2)
    if (
        coordinated_norm.item() <= float(eps)
        or not torch.isfinite(coordinated_norm)
    ):
        metadata["rule"] = "degenerate_identity"
        return g_pgc_bar, metadata

    original_sum_norm = (g_sup + g_cons).norm(p=2)
    raw_gamma = original_sum_norm / (coordinated_norm + float(eps))
    gamma = raw_gamma.clamp(min=gamma_min, max=gamma_max)
    metadata.update(
        {
            "raw_gamma": float(raw_gamma.detach().item()),
            "applied_gamma": float(gamma.detach().item()),
            "rule": "conflict_meo",
        }
    )
    return gamma * g_pgc_bar, metadata
