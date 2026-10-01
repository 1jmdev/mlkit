"""Calibration statistics and complete conversion of a synthetic block model."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

import mlkit as mk
from benchmarks.harness import Case, seeded_generator
from mlkit.calibration import StatisticAccumulator

GROUP = "calibration"


class FeedForwardBlock(nn.Module):
    """A residual expansion and contraction with the layer names of a transformer block."""

    def __init__(self, width: int, expansion: int) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(width, expansion, bias=False)
        self.mlp.down_proj = nn.Linear(expansion, width, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expanded = torch.nn.functional.silu(self.mlp.up_proj(hidden_states))
        return hidden_states + self.mlp.down_proj(expanded)


class BlockModel(nn.Module):
    def __init__(self, blocks: int, width: int, expansion: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            FeedForwardBlock(width, expansion) for _ in range(blocks)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.model.layers:
            inputs = block(inputs)
        return inputs


def statistic(name: str, rows: int, width: int) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        inputs = torch.randn(
            rows, width, device="cuda", dtype=torch.float16, generator=seeded_generator(61)
        )

        def operation() -> Any:
            accumulator = StatisticAccumulator(name)
            accumulator.update(inputs)
            return accumulator.result()

        return operation

    return prepare


def conversion(
    recipe: Callable[[], Any],
    *,
    sequential: bool = True,
) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        torch.manual_seed(67)
        model = mk.Model(BlockModel(blocks=4, width=1024, expansion=4096).cuda().half())
        generator = seeded_generator(71)
        batches = [
            torch.randn(1, 512, 1024, device="cuda", dtype=torch.float16, generator=generator)
            for _ in range(16)
        ]
        return lambda: mk.quantize(
            model, recipe(), calib=batches, sequential=sequential, cache_dir=None
        )

    return prepare


def finetuned_recipe() -> mk.Recipe:
    return mk.Recipe(
        weights=mk.gptq(mk.int(4, group=128)),
        passes=[mk.finetune(steps=8, bs=2)],
    )


def cases() -> list[Case]:
    int4 = mk.int(4, group=128)
    weights = 4 * 2 * 1024 * 4096
    definitions: list[tuple[str, Callable[[], Callable[[], Any]], int, int]] = [
        ("hessian/2048-rows-2048-wide", statistic("H", 2048, 2048), 10, 2048 * 2048),
        ("hessian/2048-rows-8192-wide", statistic("H", 2048, 8192), 10, 2048 * 8192),
        (
            "activation_absmean/2048-rows-8192-wide",
            statistic("act_absmean", 2048, 8192),
            10,
            2048 * 8192,
        ),
        ("input_sample/2048-rows-8192-wide", statistic("X", 2048, 8192), 10, 2048 * 8192),
        ("quantize/rtn-int4-4-blocks", conversion(lambda: int4), 5, weights),
        ("quantize/gptq-int4-4-blocks", conversion(lambda: mk.gptq(int4)), 3, weights),
        (
            "quantize/gptq-int4-4-blocks-nonsequential",
            conversion(lambda: mk.gptq(int4), sequential=False),
            3,
            weights,
        ),
        ("quantize/awq-int4-4-blocks", conversion(lambda: mk.awq(int4, grid=10)), 3, weights),
    ]
    selected = [
        Case(GROUP, name, prepare, warmup=1, repetitions=repetitions, elements=elements)
        for name, prepare, repetitions, elements in definitions
    ]
    selected.append(Case(
        GROUP,
        "quantize/gptq-int4-finetune-4-blocks",
        conversion(finetuned_recipe),
        warmup=1,
        repetitions=3,
        elements=weights,
        trains_parameters=True,
    ))
    return selected
