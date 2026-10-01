"""Dense and packed scalar-grid execution of single linear layers and layer stacks."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

import mlkit as mk
from benchmarks.harness import LLAMA_SHAPES, QWEN_SHAPES, Case, seeded_generator

GROUP = "inference"


def int4() -> mk.Quantizer:
    return mk.int(4, group=128)


def nf4() -> mk.Quantizer:
    return mk.nf4(group=64)


def linear(
    shape: tuple[int, int],
    batch: int,
    backend: str,
    format: Callable[[], mk.Quantizer],
) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        torch.manual_seed(41)
        output_width, input_width = shape
        module = nn.Sequential(
            nn.Linear(input_width, output_width, bias=False, device="cuda", dtype=torch.float16)
        )
        converted = mk.quantize(module, format(), calib=None)
        model = mk.optimize(converted, backend=backend, inplace=True)
        inputs = torch.randn(
            batch, input_width, device="cuda", dtype=torch.float16, generator=seeded_generator(79)
        )
        layer = model.module[0]
        return lambda: layer(inputs)

    return prepare


def layer_stack(backend: str, batch: int) -> Callable[[], Callable[[], Any]]:
    """Decoded tokens through the projections of sixteen Llama-sized blocks.

    The stack holds 1.9 GiB of FP16 weights, far more than the GPU cache, so
    every layer is read from memory as it is during generation.
    """
    def prepare() -> Callable[[], Any]:
        torch.manual_seed(41)
        layers = []
        for _ in range(16):
            for output_width, input_width in [(2048, 2048), (8192, 2048), (2048, 8192)]:
                layers.append(nn.Linear(
                    input_width, output_width, bias=False, device="cuda", dtype=torch.float16
                ))
        converted = mk.quantize(nn.Sequential(*layers), int4(), calib=None)
        model = mk.optimize(converted, backend=backend, inplace=True)
        inputs = torch.randn(
            batch, 2048, device="cuda", dtype=torch.float16, generator=seeded_generator(79)
        )
        stack = model.module
        return lambda: stack(inputs)

    return prepare


def linear_case(
    format_name: str,
    format: Callable[[], mk.Quantizer],
    model_name: str,
    label: str,
    shape: tuple[int, int],
    batch: int,
    backend: str,
) -> Case:
    return Case(
        GROUP,
        f"linear-{format_name}/{backend}-{model_name}-{label}-{shape[0]}x{shape[1]}-batch{batch}",
        linear(shape, batch, backend, format),
        warmup=20,
        repetitions=100,
        elements=shape[0] * shape[1],
        parameters={"batch": batch, "backend": backend},
    )


def cases() -> list[Case]:
    selected: list[Case] = []
    for label, shape in LLAMA_SHAPES.items():
        for batch in (1, 4, 128):
            for backend in ("dense", "packed"):
                selected.append(linear_case("int4", int4, "llama", label, shape, batch, backend))
    for label, shape in QWEN_SHAPES.items():
        selected.append(linear_case("int4", int4, "qwen", label, shape, 1, "packed"))
    for label in ("attention", "contraction"):
        shape = LLAMA_SHAPES[label]
        selected.append(linear_case("nf4", nf4, "llama", label, shape, 1, "packed"))
    stack_elements = 16 * (2048 * 2048 + 2 * 8192 * 2048)
    for batch in (1, 4, 8, 32):
        for backend in ("dense", "packed"):
            selected.append(Case(
                GROUP,
                f"layer-stack-int4/{backend}-48-layers-batch{batch}",
                layer_stack(backend, batch),
                warmup=10,
                repetitions=50,
                elements=stack_elements,
                parameters={"batch": batch, "backend": backend},
            ))
    return selected
