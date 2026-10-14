from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor


@torch.no_grad()
def decision_stability(original: Tensor, reconstructed: Tensor, top_k: int = 8) -> dict[str, Any]:
    original = original.float()
    reconstructed = reconstructed.float()
    top_k = min(top_k, original.shape[-1])
    original_selected, original_top = original.topk(top_k, dim=-1)
    reconstructed_top = reconstructed.topk(top_k, dim=-1).indices
    reconstructed_selected = reconstructed.gather(-1, original_top)
    pairs = torch.triu_indices(top_k, top_k, offset=1, device=original.device)
    margins = original_selected[..., pairs[0]] - original_selected[..., pairs[1]]
    reconstructed_margins = (
        reconstructed_selected[..., pairs[0]] - reconstructed_selected[..., pairs[1]]
    )
    flips = margins * reconstructed_margins < 0
    importance = (margins != 0) / (
        margins.abs() / original.abs().mean(-1, keepdim=True).clamp_min(1e-12) + 1e-3
    )
    stable = original.argmax(-1) == reconstructed.argmax(-1)
    measurement = {
        "local_channel_dbs": stable.float().mean(),
        "sign_flip_rate": ((original >= 0) != (reconstructed >= 0)).float().mean(),
        "top_k_overlap": (
            (original_top[..., :, None] == reconstructed_top[..., None, :])
            .any(-1)
            .float()
            .mean()
        ),
        "direction_drift": (
            1 - functional.cosine_similarity(original, reconstructed, dim=-1)
        ).mean(),
        "ordering_flip_rate": flips.sum() / (margins != 0).sum().clamp_min(1),
        "margin_weighted_decision_risk": (
            (flips * importance).sum() / importance.sum().clamp_min(1e-12)
        ),
    }
    return dict(zip(measurement, torch.stack(list(measurement.values())).cpu().tolist())) | {
        "tested_local_decisions": stable.numel(),
    }


def conversion_monitor(model, *, stability, top_k, progress):
    widths = (9, 5, 14, 9, 9, 8, 9, 11, 8, 9, 9) if stability else (9, 5, 14, 9)
    columns = (
        (
            "Progress", "Layer", "Projection", "DBS", "Avg DBS", "Sign",
            f"Top-{top_k}", "Drift", "Order", "MWDR", "Time",
        )
        if stability else ("Progress", "Layer", "Projection", "Time")
    )
    if progress:
        header = "  ".join(f"{column:>{width}}" for column, width in zip(columns, widths))
        print(f"\n{header}\n{'-' * len(header)}", flush=True)

    def observe(context, measurement, seconds):
        if stability:
            model.stability_reports[context.name] = measurement
        if progress:
            values = [str(len(model.layer_reports)), str(context.block_idx), context.name]
            if context.block is not None:
                values[2] = context.name.rsplit(".", 1)[-1]
            if stability:
                tested = sum(
                    record["tested_local_decisions"] for record in model.stability_reports.values()
                )
                unchanged = sum(
                    round(record["local_channel_dbs"] * record["tested_local_decisions"])
                    for record in model.stability_reports.values()
                )
                values.extend([
                    f"{measurement['local_channel_dbs']:.2%}",
                    f"{unchanged / tested:.2%}",
                    f"{measurement['sign_flip_rate']:.2%}",
                    f"{measurement['top_k_overlap']:.2%}",
                    f"{measurement['direction_drift']:.2e}",
                    f"{measurement['ordering_flip_rate']:.2%}",
                    f"{measurement['margin_weighted_decision_risk']:.2%}",
                ])
            values.append(f"{seconds:.3f}s")
            print(
                "  ".join(f"{value:>{width}}" for value, width in zip(values, widths)),
                flush=True,
            )

    return observe
