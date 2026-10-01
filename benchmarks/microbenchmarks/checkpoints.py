"""Checkpoint writing and reading of a converted synthetic model."""

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

import mlkit as mk
from benchmarks.harness import Case

GROUP = "checkpoints"


def create_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2048, 2048, bias=False),
        nn.Linear(2048, 2048, bias=False),
        nn.Linear(2048, 2048, bias=False),
        nn.Linear(2048, 2048, bias=False),
    )


def converted_model(format: Callable[[], Any]) -> mk.QModel:
    torch.manual_seed(73)
    return mk.quantize(create_model().cuda().half(), format(), calib=None)


def saving(format: Callable[[], Any]) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        converted = converted_model(format)
        # The temporary directory is removed when the returned operation is released.
        directory = tempfile.TemporaryDirectory(prefix="mlkit_benchmark_")
        destination = Path(directory.name) / "checkpoint"

        def operation() -> None:
            assert directory.name
            converted.save(destination, overwrite=True)

        return operation

    return prepare


def loading(format: Callable[[], Any]) -> Callable[[], Callable[[], Any]]:
    def prepare() -> Callable[[], Any]:
        directory = tempfile.TemporaryDirectory(prefix="mlkit_benchmark_")
        source = Path(directory.name) / "checkpoint"
        converted_model(format).save(source)

        def operation() -> Any:
            assert directory.name
            return mk.load_checkpoint(source, model=create_model)

        return operation

    return prepare


def cases() -> list[Case]:
    elements = 4 * 2048 * 2048
    definitions: list[tuple[str, Callable[[], Callable[[], Any]]]] = [
        ("save/int4-g128-4x2048x2048", saving(lambda: mk.int(4, group=128))),
        ("load/int4-g128-4x2048x2048", loading(lambda: mk.int(4, group=128))),
        ("save/int3-g128-4x2048x2048", saving(lambda: mk.int(3, group=128))),
        ("load/int3-g128-4x2048x2048", loading(lambda: mk.int(3, group=128))),
    ]
    return [
        Case(GROUP, name, prepare, warmup=1, repetitions=5, elements=elements)
        for name, prepare in definitions
    ]
