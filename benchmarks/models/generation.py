"""Measure complete greedy generation, including prompt processing and KV updates.

    uv run python -m benchmarks.models.generation --model Qwen/Qwen2.5-0.5B

Every variant is built and warmed first. The timed generations are then
interleaved: each round runs every variant once, starting from a different
variant, so a drift of the GPU clock state affects all variants alike. Speedups
are paired within a round against the first variant and summarized by their
median.
"""

import argparse
import copy
import json
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch

import mlkit as mk

PROMPT = "Explain the difference between calibration and inference in weight quantization."
VARIANTS = [
    "dense",
    "dense-compiled",
    "packed",
    "packed-compiled",
    "packed-head",
    "packed-head-compiled",
    "torchao-int4",
    "torchao-int4-compiled",
    "torchao-int8",
    "torchao-int8-compiled",
]
DEFAULT_VARIANTS = VARIANTS[:6]


def sharing_copy(model: mk.Model) -> mk.Model:
    """Copy the modules of ``model`` while sharing every parameter and buffer."""
    tensors = [*model.module.parameters(), *model.module.buffers()]
    return copy.deepcopy(model, {id(tensor): tensor for tensor in tensors})


def build_variants(arguments: argparse.Namespace) -> dict[str, mk.Model]:
    """Build the requested variants; compiled variants share the weights of eager ones."""
    selected: dict[str, mk.Model] = {}
    requested = set(arguments.variants)

    def add(name: str, eager: Callable[[], mk.Model]) -> None:
        compiled_name = f"{name}-compiled"
        if name not in requested and compiled_name not in requested:
            return
        model = eager()
        if name in requested:
            selected[name] = model
        if compiled_name in requested:
            selected[compiled_name] = mk.optimize(
                sharing_copy(model), backend="dense", compile=True, inplace=True
            )
            selected[compiled_name].execution_backend = model.execution_backend + "+compiled"

    source = mk.load(arguments.model, dtype="float16")
    blocks = mk.int(4, group=arguments.group)
    head = mk.int(arguments.head_bits, group=arguments.group)
    add("dense", lambda: source)
    add("packed", lambda: mk.optimize(
        mk.quantize(source, blocks, calib=None), backend="packed", inplace=True
    ))
    add("packed-head", lambda: mk.optimize(
        mk.quantize(source, mk.Recipe(weights=blocks, head=head), calib=None),
        backend="packed",
        inplace=True,
    ))
    for bits in (4, 8):
        add(f"torchao-int{bits}", lambda bits=bits: mk.export_torchao(source, bits=bits))
    return {name: selected[name] for name in arguments.variants}


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--head-bits", type=int, default=8)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=DEFAULT_VARIANTS)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/generation.json"))
    arguments = parser.parse_args()
    variants = build_variants(arguments)
    tokenizer = next(iter(variants.values())).tokenizer
    inputs = tokenizer(PROMPT, return_tensors="pt")
    prompt_tokens = inputs["input_ids"].numel()
    generators = {
        name: partial(
            model.generate,
            **inputs,
            do_sample=False,
            min_new_tokens=arguments.tokens,
            max_new_tokens=arguments.tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        for name, model in variants.items()
    }
    generated: dict[str, list[int]] = {}
    with torch.inference_mode():
        for name, generate in generators.items():
            for _ in range(arguments.warmup):
                generate()
            generated[name] = generate()[0, prompt_tokens:].tolist()
            print(f"warmed {name}", flush=True)
        names = list(generators)
        samples: dict[str, list[float]] = {name: [] for name in names}
        for round_index in range(arguments.rounds):
            start_position = round_index % len(names)
            for name in names[start_position:] + names[:start_position]:
                torch.cuda.synchronize()
                start = time.perf_counter()
                generators[name]()
                torch.cuda.synchronize()
                samples[name].append(time.perf_counter() - start)
    reference = names[0]
    records: list[dict[str, Any]] = []
    for name in names:
        durations = samples[name]
        paired = [
            reference_duration / duration
            for reference_duration, duration in zip(samples[reference], durations, strict=True)
        ]
        record = {
            "variant": name,
            "backend": variants[name].execution_backend,
            "tokens_per_second": arguments.tokens / statistics.median(durations),
            "fastest_tokens_per_second": arguments.tokens / min(durations),
            "slowest_tokens_per_second": arguments.tokens / max(durations),
            "paired_speedup": statistics.median(paired),
            "model_storage_bytes": variants[name].storage_bytes,
            "matches_reference_tokens": generated[name] == generated[reference],
            "generated_ids": generated[name],
            "seconds": durations,
        }
        records.append(record)
        slowest = record["slowest_tokens_per_second"]
        fastest = record["fastest_tokens_per_second"]
        print(
            f"{name:24} {record['tokens_per_second']:8.1f} tokens/s"
            f"  [{slowest:.1f}, {fastest:.1f}]"
            f"  {record['paired_speedup']:5.2f}x {reference}"
            f"  {record['model_storage_bytes'] / 2**20:8.0f} MiB",
            flush=True,
        )
    document = {
        "model": arguments.model,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "dtype": "float16",
        "prompt_tokens": prompt_tokens,
        "generated_tokens": arguments.tokens,
        "rounds": arguments.rounds,
        "warmup_generations": arguments.warmup,
        "block_quantization": {"bits": 4, "group": arguments.group},
        "head_quantization": {"bits": arguments.head_bits, "group": arguments.group},
        "reference_variant": reference,
        "measurements": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
