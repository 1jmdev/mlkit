"""Measure deterministic weighted codebook fitting on CUDA samples."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch

import mlkit as mk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--dimension", type=int, default=1)
    parser.add_argument("--centers", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_results/codebook_fitting.json")
    )
    arguments = parser.parse_args()
    generator = torch.Generator(device="cuda").manual_seed(17)
    samples = torch.randn(
        arguments.samples, arguments.dimension, device="cuda", dtype=torch.float32,
        generator=generator,
    )
    if arguments.dimension == 1:
        samples = samples[:, 0]
    weights = torch.rand(arguments.samples, device="cuda", generator=generator)
    fit = partial(
        mk.kmeans, samples, k=arguments.centers, weights=weights,
        iters=arguments.iterations, seed=17,
    )
    measurement = mk.benchmark(fit, warmup=1, repetitions=arguments.repetitions)
    record = {
        "samples": arguments.samples,
        "dimension": arguments.dimension,
        "centers": arguments.centers,
        "iterations": arguments.iterations,
        "seed": 17,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        **measurement.to_dict(),
    }
    print(json.dumps(record), flush=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
