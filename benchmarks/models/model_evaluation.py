"""Evaluate a downloaded causal model with an explicit reproducible token budget.

    uv run python -m benchmarks.models.model_evaluation --model meta-llama/Llama-3.2-1B
"""

import argparse
import gc
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

import mlkit as mk

UNLIMITED_TOKEN_ENERGY = "gptq-int4-g128-unlimited-token-energy"


def methods() -> dict[str, Callable[[], Any]]:
    int4 = mk.int(4, group=128)
    return {
        "rtn-int4-g128": lambda: int4,
        "nf4-g64": lambda: mk.nf4(group=64),
        "gptq-int4-g128": lambda: mk.gptq(int4),
        UNLIMITED_TOKEN_ENERGY: lambda: mk.gptq(int4),
        "gptq-nf4-g64": lambda: mk.gptq(mk.nf4(group=64)),
        "gptq-int4-g128-activation-order": lambda: mk.gptq(int4, act_order=True),
        "gptq-int4-g128+head-int8": lambda: mk.Recipe(
            weights=mk.gptq(int4), head=mk.int(8, group=128)
        ),
        "gptq-int4-g128+head-gptq-int4": lambda: mk.Recipe(
            weights=mk.gptq(int4), head=mk.gptq(int4)
        ),
    }


def main() -> None:
    available = methods()
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--evaluation-batches", type=int, default=40)
    parser.add_argument("--full-evaluation", action="store_true")
    parser.add_argument("--methods", nargs="+", default=list(available), choices=list(available))
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_results/model_evaluation.json")
    )
    arguments = parser.parse_args()
    model = mk.load(arguments.model, dtype="float16")
    calibration = mk.data(
        "wikitext2",
        n=arguments.calibration_batches,
        seq=arguments.sequence,
        tokenizer=model.tokenizer,
        seed=17,
    )
    evaluation_batches = None if arguments.full_evaluation else arguments.evaluation_batches
    evaluation = mk.data(
        "wikitext2",
        n=evaluation_batches,
        seq=arguments.sequence,
        tokenizer=model.tokenizer,
        split="test",
        seed=17,
    )
    baseline = mk.ppl(
        model, data=evaluation, budget="full", seq=arguments.sequence, return_details=True
    )
    records: list[dict[str, Any]] = [{
        "method": "baseline",
        "perplexity": baseline.perplexity,
        "tokens": baseline.tokens,
        "seconds": baseline.seconds,
    }]
    print(json.dumps(records[-1]), flush=True)
    for name in arguments.methods:
        torch.cuda.synchronize()
        start = time.perf_counter()
        converted = mk.quantize(
            model,
            available[name](),
            calib=calibration,
            cache_dir=None,
            token_energy_limit=None if name == UNLIMITED_TOKEN_ENERGY else 100.0,
        )
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        score = mk.ppl(
            converted, data=evaluation, budget="full", seq=arguments.sequence, return_details=True
        )
        records.append({
            "method": name,
            "bpw": converted.bpw,
            "model_bpw": converted.model_bpw,
            "perplexity": score.perplexity,
            "tokens": score.tokens,
            "quantization_seconds": duration,
            "evaluation_seconds": score.seconds,
        })
        print(json.dumps(records[-1]), flush=True)
        del converted
        gc.collect()
        torch.cuda.empty_cache()
    document = {
        "model": arguments.model,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "dtype": "float16",
        "sequence": arguments.sequence,
        "seed": 17,
        "calibration_batches": arguments.calibration_batches,
        "evaluation_batches": evaluation_batches,
        "measurements": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
