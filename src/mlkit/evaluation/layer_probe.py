"""Hessian proxy loss of candidate quantizers on selected layers, without conversion."""

import fnmatch
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mlkit.calibration.session import CalibrationSession
from mlkit.calibration.statistics import BlockStatistics
from mlkit.evaluation.tables import Table
from mlkit.models.model import Model
from mlkit.quantization.context import Ctx
from mlkit.quantization.operations.losses import proxy_loss
from mlkit.quantization.representation import as_q
from mlkit.timing import synchronize


def probe(
    quantizers: Sequence[Callable],
    model: Model | nn.Module,
    layers: str = "*.mlp.down_proj",
    *,
    calib: Any = "c4",
    cache_dir: str | Path | None = "~/.cache/mlkit/statistics",
    seed: int = 0,
    print_table: bool = True,
    token_energy_limit: float | None = 100.0,
) -> Table:
    wrapped = model if isinstance(model, Model) else Model(model)
    selected_blocks = tuple(
        index for index, block in enumerate(wrapped.blocks)
        if any(
            isinstance(module, nn.Linear) and fnmatch.fnmatchcase(
                f"{wrapped.architecture.block_name(index)}.{name}".strip("."), layers
            )
            for name, module in block.named_modules()
        )
    )
    session = CalibrationSession(
        wrapped, calib, sequential=False, sample_rows=4096,
        cache_dir=None if cache_dir is None else Path(cache_dir).expanduser(), need_targets=False,
        selected_blocks=selected_blocks,
        token_energy_limit=token_energy_limit,
    )
    records = []
    for index, block in enumerate(wrapped.blocks):
        prefix = wrapped.architecture.block_name(index)
        statistics = BlockStatistics(session, block, index, prefix)
        for relative_name, module in block.named_modules():
            name = f"{prefix}.{relative_name}".strip(".")
            if not isinstance(module, nn.Linear) or not fnmatch.fnmatchcase(name, layers):
                continue
            weight = module.weight.detach().float()
            for algorithm in quantizers:
                context = Ctx(name, module, block, index, seed=seed, device=weight.device,
                              provider=statistics.provider(name, weight.device))
                synchronize(weight.device)
                start = time.perf_counter()
                with torch.no_grad():
                    quantized = as_q(algorithm(weight, context))
                    loss = float(proxy_loss(weight, quantized.w, context))
                synchronize(weight.device)
                bits = None if quantized.bits is None else quantized.bits + context._additional_bits
                records.append({"layer": name, "method": repr(algorithm),
                                "bpw": None if bits is None else bits / weight.numel(),
                                "loss": loss, "seconds": time.perf_counter() - start})
    result = Table(records)
    if print_table:
        print(result)
    return result
