"""Calibrated algorithms applied to one weight matrix with a fixed Hessian."""

from collections.abc import Callable
from typing import Any

import torch

import mlkit as mk
from benchmarks.harness import (
    LLAMA_SHAPES,
    QWEN_SHAPES,
    Case,
    seeded_generator,
    synthetic_inputs,
    synthetic_weight,
)

GROUP = "algorithms"


def calibrated(
    algorithm: Callable[[], mk.Quantizer],
    shape: tuple[int, int],
    *,
    name: str = "layer",
) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        weight = synthetic_weight(shape)
        inputs = synthetic_inputs(shape[1])
        hessian = inputs.T @ inputs / len(inputs)
        activation_mean = inputs.abs().mean(0)
        quantizer = algorithm()
        del inputs

        def operation() -> Any:
            context = mk.Ctx(name, H=hessian, stats={"act_absmean": activation_mean})
            return quantizer(weight, context).w

        return operation

    return prepare


def vector_feedback() -> mk.Quantizer:
    codebook = torch.randn(256, 8, device="cuda", generator=seeded_generator(47))
    return mk.ldlq(mk.scaled(mk.grid.vector(codebook), group=None), step=8, refit=None)


def cases() -> list[Case]:
    int4 = mk.int(4, group=128)
    selected: list[tuple[str, Callable[[], mk.Quantizer], tuple[int, int], int]] = []
    for label, shape in LLAMA_SHAPES.items():
        selected.append((f"gptq-int4/llama-{label}", lambda: mk.gptq(int4), shape, 5))
    for label, shape in QWEN_SHAPES.items():
        selected.append((f"gptq-int4/qwen-{label}", lambda: mk.gptq(int4), shape, 5))
    attention = LLAMA_SHAPES["attention"]
    contraction = LLAMA_SHAPES["contraction"]
    key_value = LLAMA_SHAPES["key_value"]
    selected.extend([
        ("gptq-int4-torch-reference/llama-key_value",
         lambda: mk.gptq(int4, backend="torch"), key_value, 3),
        ("gptq-int4-activation-order/llama-attention",
         lambda: mk.gptq(int4, act_order=True), attention, 5),
        ("gptq-int4-activation-order/llama-contraction",
         lambda: mk.gptq(int4, act_order=True), contraction, 3),
        ("gptq-int4-whole-row-refit/llama-attention",
         lambda: mk.gptq(mk.int(4, group=None), refit=None), attention, 5),
        ("gptq-nf4/llama-attention", lambda: mk.gptq(mk.nf4(group=64)), attention, 5),
        ("gptq-int4-asymmetric/llama-attention",
         lambda: mk.gptq(mk.int(4, group=128, asym=True)), attention, 5),
        ("gptq-int3/llama-attention", lambda: mk.gptq(mk.int(3, group=128)), attention, 5),
        ("ldlq-e8p-step8/llama-attention",
         lambda: mk.ldlq(mk.scaled(mk.grid.e8p(), group=None), step=8, refit=None), attention, 3),
        ("ldlq-vector8-256-step8/llama-attention", vector_feedback, attention, 3),
        ("ldlq-e8p-step16-reference/llama-key_value",
         lambda: mk.ldlq(mk.scaled(mk.grid.e8p(), group=None), step=16, refit=None),
         key_value, 3),
        ("ldlq-trellis-l8-step16/llama-key_value",
         lambda: mk.ldlq(mk.trellis(L=8), step=16, refit=None), key_value, 3),
        ("awq-int4-grid20/llama-attention", lambda: mk.awq(int4, grid=20), attention, 3),
        ("awq-gptq-int4-grid10/llama-key_value",
         lambda: mk.awq(mk.gptq(int4), grid=10), key_value, 3),
        ("incoherent-gptq-int4/llama-attention",
         lambda: mk.incoherent(mk.gptq(int4)), attention, 3),
        ("incoherent-gptq-int4/qwen-contraction",
         lambda: mk.incoherent(mk.gptq(int4)), QWEN_SHAPES["contraction"], 3),
        ("best-of-int4-nf4/llama-attention",
         lambda: mk.best_of(int4, mk.nf4(group=64)), attention, 5),
    ])
    return [
        Case(
            GROUP,
            f"{name}-{shape[0]}x{shape[1]}",
            calibrated(algorithm, shape),
            warmup=1,
            repetitions=repetitions,
            elements=shape[0] * shape[1],
        )
        for name, algorithm, shape, repetitions in selected
    ]
