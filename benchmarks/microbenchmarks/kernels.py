"""Fused CUDA kernels measured through their public entry points."""

from collections.abc import Callable
from typing import Any

import torch

import mlkit as mk
from benchmarks.harness import Case, seeded_generator, synthetic_weight
from mlkit.quantization.grids.lattice import nearest_e8p

GROUP = "kernels"


def activation_reconstruction(
    format: Callable[[], Any],
    rows: int,
    width: int,
) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        inputs = torch.randn(
            rows, width, device="cuda", dtype=torch.float16, generator=seeded_generator(47)
        )
        quantizer = format()
        return lambda: quantizer.reconstruct_activations(inputs)

    return prepare


def lattice_search(vectors: int, backend: str) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        values = torch.randn(vectors, 8, device="cuda", generator=seeded_generator(53))
        return lambda: nearest_e8p(values, return_indices=True, backend=backend)

    return prepare


def viterbi_search(
    sequences: int,
    length: int,
    L: int,
    backend: str,
) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        values = torch.randn(sequences, length, device="cuda", generator=seeded_generator(59))
        codes = mk.one_mad(torch.arange(2**L, device="cuda"))
        return lambda: mk.viterbi(values, codes, L, 2, backend=backend, return_states=True)

    return prepare


def trellis_decode(shape: tuple[int, int]) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        quantized = mk.trellis(L=8, tile=16)(synthetic_weight(shape))
        return lambda: quantized.w

    return prepare


def scalar_decode(shape: tuple[int, int]) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        quantized = mk.int(4, group=128)(synthetic_weight(shape))
        return lambda: quantized.w

    return prepare


def cases() -> list[Case]:
    activations = 4096 * 2048
    definitions: list[tuple[str, Callable[[], Callable[[], Any]], int, int]] = [
        (
            "activation_reconstruction/int4-row-4096x2048",
            activation_reconstruction(lambda: mk.int(4, group=None), 4096, 2048),
            10,
            activations,
        ),
        (
            "activation_reconstruction/int4-g64-4096x2048",
            activation_reconstruction(lambda: mk.int(4, group=64), 4096, 2048),
            10,
            activations,
        ),
        (
            "activation_reconstruction/nf4-g64-4096x2048",
            activation_reconstruction(lambda: mk.nf4(group=64), 4096, 2048),
            10,
            activations,
        ),
        (
            "activation_reconstruction/int4-row-1x2048",
            activation_reconstruction(lambda: mk.int(4, group=None), 1, 2048),
            50,
            2048,
        ),
        ("lattice_search/e8p-2m-vectors", lattice_search(2_097_152, "triton"), 10, 2_097_152),
        (
            "lattice_search/e8p-torch-reference-65k-vectors",
            lattice_search(65_536, "torch"),
            3,
            65_536,
        ),
        (
            "viterbi_search/l8-4096-sequences-256",
            viterbi_search(4096, 256, 8, "triton"),
            5,
            4096 * 256,
        ),
        (
            "viterbi_search/l12-512-sequences-256",
            viterbi_search(512, 256, 12, "triton"),
            3,
            512 * 256,
        ),
        (
            "viterbi_search/l8-torch-reference-256-sequences-256",
            viterbi_search(256, 256, 8, "torch"),
            3,
            256 * 256,
        ),
        ("trellis_decode/l8-2048x2048", trellis_decode((2048, 2048)), 10, 2048 * 2048),
        ("scalar_decode/int4-g128-2048x8192", scalar_decode((2048, 8192)), 10, 2048 * 8192),
    ]
    return [
        Case(GROUP, name, prepare, warmup=1, repetitions=repetitions, elements=elements)
        for name, prepare, repetitions, elements in definitions
    ]
