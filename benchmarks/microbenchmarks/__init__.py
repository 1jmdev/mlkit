"""Synthetic workloads that measure every mlkit subsystem without downloading a model."""

from benchmarks.harness import Case
from benchmarks.microbenchmarks import (
    algorithms,
    calibration,
    checkpoints,
    formats,
    inference,
    kernels,
    operations,
)

GROUPS = (operations, formats, kernels, algorithms, calibration, checkpoints, inference)


def all_cases() -> list[Case]:
    return [case for group in GROUPS for case in group.cases()]
