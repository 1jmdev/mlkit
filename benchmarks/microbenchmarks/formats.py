"""Direct rounding of one weight matrix by every built-in format, including decoding."""

from collections.abc import Callable
from typing import Any

import torch

import mlkit as mk
from benchmarks.harness import Case, seeded_generator, synthetic_weight

GROUP = "formats"


def rounding(
    format: Callable[[], mk.Quantizer],
    shape: tuple[int, int],
) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        weight = synthetic_weight(shape)
        quantizer = format()
        return lambda: quantizer(weight).w

    return prepare


def vector_codebook() -> mk.Quantizer:
    codebook = torch.randn(256, 8, device="cuda", generator=seeded_generator(43))
    return mk.scaled(mk.grid.vector(codebook), group=None)


def cases() -> list[Case]:
    large = (2048, 8192)
    medium = (2048, 2048)
    small = (512, 2048)
    formats: list[tuple[str, Callable[[], mk.Quantizer], tuple[int, int], int]] = [
        ("int4-g128", lambda: mk.int(4, group=128), large, 10),
        ("int4-row", lambda: mk.int(4, group=None), large, 10),
        ("int3-g128", lambda: mk.int(3, group=128), large, 10),
        ("int8-g128", lambda: mk.int(8, group=128), large, 10),
        ("int4-asymmetric-g128", lambda: mk.int(4, group=128, asym=True), large, 10),
        ("int4-mse-g128", lambda: mk.int(4, group=128, scale="mse"), large, 5),
        ("nf4-g64", lambda: mk.nf4(group=64), large, 10),
        ("mxfp4-g32", lambda: mk.mxfp4(), large, 10),
        ("fp8-e4m3-g128", lambda: mk.scaled(mk.grid.fp("e4m3"), group=128), large, 10),
        ("vector8-256-row", vector_codebook, medium, 5),
        ("e8p-row", lambda: mk.scaled(mk.grid.e8p(), group=None), large, 5),
        ("trellis-l8-tile16", lambda: mk.trellis(L=8, tile=16), medium, 3),
        ("trellis-l12-tile16", lambda: mk.trellis(L=12, tile=16), small, 3),
    ]
    return [
        Case(
            GROUP,
            f"{name}/{shape[0]}x{shape[1]}",
            rounding(format, shape),
            warmup=1,
            repetitions=repetitions,
            elements=shape[0] * shape[1],
        )
        for name, format, shape, repetitions in formats
    ]
