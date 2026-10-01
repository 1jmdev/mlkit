"""Numerical building blocks: clustering, search, orthogonal transforms and bit packing."""

from collections.abc import Callable
from typing import Any

import torch

import mlkit as mk
from benchmarks.harness import Case, seeded_generator, synthetic_hessian, synthetic_weight
from mlkit.quantization.operations import structured_transform

GROUP = "operations"


def scalar_clustering() -> Callable[[], Any]:
    generator = seeded_generator(17)
    samples = torch.randn(1_000_000, device="cuda", generator=generator)
    weights = torch.rand(1_000_000, device="cuda", generator=generator)
    return lambda: mk.kmeans(samples, k=16, weights=weights, iters=10, seed=17)


def vector_clustering() -> Callable[[], Any]:
    generator = seeded_generator(17)
    samples = torch.randn(262_144, 8, device="cuda", generator=generator)
    weights = torch.rand(262_144, device="cuda", generator=generator)
    return lambda: mk.kmeans(samples, k=256, weights=weights, iters=5, seed=17)


def vector_search(samples: int, codewords: int) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        generator = seeded_generator(31)
        values = torch.randn(samples, 8, device="cuda", generator=generator)
        codebook = torch.randn(codewords, 8, device="cuda", generator=generator)
        return lambda: mk.nearest(values, codebook, return_indices=True)

    return prepare


def scalar_snap() -> Callable[[], Any]:
    values = torch.randn(2048, 8192, device="cuda", generator=seeded_generator(31))
    codebook = torch.linspace(-3, 3, 16, device="cuda")
    return lambda: mk.snap(values, codebook)


def transform(width: int, rows: int) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        values = torch.randn(rows, width, device="cuda", generator=seeded_generator(37))
        return lambda: structured_transform(values)

    return prepare


def randomized_transform() -> Callable[[], Any]:
    values = torch.randn(2048, 8192, device="cuda", generator=seeded_generator(37))
    return lambda: mk.rht(values, seed=7)


def pack_codes(bits: int) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        codes = torch.randint(
            2**bits, (2048, 8192), device="cuda", dtype=torch.uint8, generator=seeded_generator(41)
        )
        return lambda: mk.pack(codes, bits)

    return prepare


def unpack_codes(bits: int) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        codes = torch.randint(
            2**bits, (2048, 8192), device="cuda", dtype=torch.uint8, generator=seeded_generator(41)
        )
        packed = mk.pack(codes, bits)
        return lambda: mk.unpack(packed, bits, (2048, 8192))

    return prepare


def reconstruction_loss() -> Callable[[], Any]:
    weight = synthetic_weight((2048, 8192))
    reconstruction = weight + 0.001 * torch.randn_like(weight)
    context = mk.Ctx(H=synthetic_hessian(8192))
    return lambda: mk.proxy_loss(weight, reconstruction, context)


def cases() -> list[Case]:
    matrix = 2048 * 8192
    definitions: list[tuple[str, Callable[[], Callable[[], Any]], int, int]] = [
        ("kmeans/scalar-1m-k16-10iters", scalar_clustering, 5, 10 * 1_000_000),
        ("kmeans/vector8-262k-k256-5iters", vector_clustering, 5, 5 * 262_144),
        ("nearest/1m-vectors-256-codewords", vector_search(1_000_000, 256), 5, 1_000_000),
        ("nearest/65k-vectors-65k-codewords", vector_search(65_536, 65_536), 3, 65_536),
        ("snap/2048x8192-16-values", scalar_snap, 10, matrix),
        ("structured_transform/2048x8192", transform(8192, 2048), 10, matrix),
        ("structured_transform/4864x896", transform(896, 4864), 10, 4864 * 896),
        ("structured_transform/896x4864", transform(4864, 896), 10, 4864 * 896),
        ("rht/2048x8192", randomized_transform, 10, matrix),
        ("pack/4bit-2048x8192", pack_codes(4), 10, matrix),
        ("pack/3bit-2048x8192", pack_codes(3), 10, matrix),
        ("unpack/4bit-2048x8192", unpack_codes(4), 10, matrix),
        ("unpack/3bit-2048x8192", unpack_codes(3), 10, matrix),
        ("proxy_loss/2048x8192", reconstruction_loss, 10, matrix),
    ]
    return [
        Case(GROUP, name, prepare, warmup=1, repetitions=repetitions, elements=elements)
        for name, prepare, repetitions, elements in definitions
    ]
