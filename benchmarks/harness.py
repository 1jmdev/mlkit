"""Shared measurement protocol, synthetic workloads and result files for benchmarks."""

import datetime
import fnmatch
import gc
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import triton
from torch import Tensor

import mlkit as mk

RESULTS_DIRECTORY = Path("benchmark_results")

# Linear layer shapes [out, in] of the two development models.
LLAMA_SHAPES = {
    "attention": (2048, 2048),
    "key_value": (512, 2048),
    "expansion": (8192, 2048),
    "contraction": (2048, 8192),
}
QWEN_SHAPES = {
    "attention": (896, 896),
    "key_value": (128, 896),
    "expansion": (4864, 896),
    "contraction": (896, 4864),
}


@dataclass
class Case:
    """One measured operation.

    ``prepare`` allocates inputs and returns the operation to time, so allocation
    and data generation are excluded from every measurement.
    """

    group: str
    name: str
    prepare: Callable[[], Callable[[], Any]]
    warmup: int = 2
    repetitions: int = 10
    elements: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    trains_parameters: bool = False

    @property
    def identifier(self) -> str:
        return f"{self.group}/{self.name}"


def environment() -> dict[str, Any]:
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "measurement_date": datetime.date.today().isoformat(),
    }


def release_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def seeded_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def synthetic_weight(shape: tuple[int, int], seed: int = 17) -> Tensor:
    """A weight matrix with the heavy-tailed rows and columns of trained projections."""
    generator = seeded_generator(seed)
    weight = torch.randn(shape, device="cuda", generator=generator)
    row_scales = torch.rand(shape[0], 1, device="cuda", generator=generator) + 0.5
    column_scales = torch.rand(1, shape[1], device="cuda", generator=generator) + 0.5
    return weight * row_scales * column_scales * 0.02


def synthetic_inputs(width: int, rows: int = 4096, seed: int = 29) -> Tensor:
    """Correlated layer inputs with unequal channel magnitudes."""
    generator = seeded_generator(seed)
    independent = torch.randn(rows, width, device="cuda", generator=generator)
    shared = torch.randn(rows, 16, device="cuda", generator=generator)
    mixing = torch.randn(16, width, device="cuda", generator=generator)
    channel_scales = torch.rand(width, device="cuda", generator=generator) * 2 + 0.1
    return (independent + 0.5 * shared @ mixing) * channel_scales


def synthetic_hessian(width: int, rows: int = 4096, seed: int = 29) -> Tensor:
    inputs = synthetic_inputs(width, rows, seed)
    return inputs.T @ inputs / rows


def measure(case: Case) -> dict[str, Any]:
    """Time the first call separately, then report synchronized steady-state latency."""
    release_memory()
    operation = case.prepare()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    inference_mode = not case.trains_parameters
    start = time.perf_counter()
    with torch.inference_mode(inference_mode):
        operation()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - start) * 1000
    measurement = mk.benchmark(
        operation,
        warmup=case.warmup,
        repetitions=case.repetitions,
        inference_mode=inference_mode,
    )
    record: dict[str, Any] = {
        "case": case.identifier,
        "median_ms": measurement.median_ms,
        "minimum_ms": measurement.minimum_ms,
        "percentile_95_ms": measurement.percentile_95_ms,
        "first_call_ms": first_call_ms,
        "repetitions": measurement.repetitions,
        "additional_peak_memory_bytes": max(0, (measurement.peak_memory_bytes or 0) - resident),
        "parameters": case.parameters,
    }
    if case.elements is not None:
        record["elements_per_second"] = case.elements / (measurement.median_ms / 1000)
    del operation
    release_memory()
    return record


def select(cases: Iterable[Case], patterns: list[str] | None) -> list[Case]:
    if not patterns:
        return list(cases)
    return [
        case for case in cases
        if any(fnmatch.fnmatchcase(case.identifier, pattern) for pattern in patterns)
    ]


def run(cases: Iterable[Case]) -> list[dict[str, Any]]:
    records = []
    for case in cases:
        record = measure(case)
        records.append(record)
        print(format_record(record), flush=True)
    return records


def format_record(record: dict[str, Any]) -> str:
    throughput = record.get("elements_per_second")
    rate = "" if throughput is None else f"  {throughput / 1e6:10.1f} M elements/s"
    return (
        f"{record['case']:58} {record['median_ms']:10.3f} ms"
        f"  first {record['first_call_ms']:10.1f} ms"
        f"  +{record['additional_peak_memory_bytes'] / 2**20:8.1f} MiB{rate}"
    )


def write_results(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(environment() | document, indent=2) + "\n")
