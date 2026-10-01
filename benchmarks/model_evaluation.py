"""Evaluate a downloaded causal model with an explicit reproducible token budget."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

import mlkit as mk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--evaluation-batches", type=int, default=16)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_results/model_evaluation.json")
    )
    arguments = parser.parse_args()
    model = mk.load(arguments.model, dtype="float16")
    calibration = mk.data("wikitext2", n=arguments.calibration_batches, seq=arguments.sequence,
                          tokenizer=model.tokenizer, seed=17)
    evaluation = mk.data("wikitext2", n=arguments.evaluation_batches, seq=arguments.sequence,
                         tokenizer=model.tokenizer, split="test", seed=17)
    baseline = mk.ppl(model, data=evaluation, budget="full", seq=arguments.sequence,
                      return_details=True)
    records = [{"method": "baseline", "perplexity": baseline.perplexity,
                "tokens": baseline.tokens, "seconds": baseline.seconds}]
    print(json.dumps(records[-1]), flush=True)
    methods = {
        "rtn-int4-g128": mk.int(4, group=128),
        "nf4-g64": mk.nf4(group=64),
        "gptq-int4-g128": mk.gptq(mk.int(4, group=128)),
    }
    for name, quantization in methods.items():
        start = time.perf_counter()
        converted = mk.quantize(model, quantization, calib=calibration, cache_dir=None)
        duration = time.perf_counter() - start
        score = mk.ppl(converted, data=evaluation, budget="full", seq=arguments.sequence,
                       return_details=True)
        records.append({"method": name, "bpw": converted.bpw,
                        "perplexity": score.perplexity, "tokens": score.tokens,
                        "quantization_seconds": duration, "evaluation_seconds": score.seconds})
        print(json.dumps(records[-1]), flush=True)
        del converted
        gc.collect()
    document = {"model": arguments.model, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(), "sequence": arguments.sequence,
                "cuda": torch.version.cuda, "dtype": "float16", "seed": 17,
                "calibration_batches": arguments.calibration_batches,
                "evaluation_batches": arguments.evaluation_batches, "measurements": records}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
