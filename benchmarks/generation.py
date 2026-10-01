"""Measure complete greedy generation, including prompt processing and KV updates."""

import argparse
import gc
import json
from pathlib import Path

import torch

import mlkit as mk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/generation.json"))
    arguments = parser.parse_args()
    model = mk.load(arguments.model, dtype="float16")
    inputs = model.tokenizer(
        "Explain the difference between calibration and inference in weight quantization.",
        return_tensors="pt",
    )
    records = []
    quantized = None
    for backend in ["baseline", "dense", "packed"]:
        if backend == "baseline":
            selected = model
        else:
            if quantized is None:
                quantized = mk.quantize(model, mk.int(4, group=128), calib=None)
            selected = mk.optimize(quantized, backend=backend, compile=arguments.compile)

        def generate(selected=selected):
            return selected.generate(
                **inputs, do_sample=False, min_new_tokens=arguments.tokens,
                max_new_tokens=arguments.tokens, pad_token_id=model.tokenizer.eos_token_id,
            )

        measurement = mk.benchmark(generate, repetitions=arguments.repetitions, warmup=2)
        output = generate()
        record = {
            "backend": selected.execution_backend, "method": backend,
            "prompt_tokens": inputs["input_ids"].numel(), "generated_tokens": arguments.tokens,
            "tokens_per_second": arguments.tokens * 1000 / measurement.median_ms,
            "generated_ids": output[0, inputs["input_ids"].shape[1] :].tolist(),
            **measurement.to_dict(),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        if backend != "baseline":
            del selected
            gc.collect()
            torch.cuda.empty_cache()
    document = {
        "model": arguments.model, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "compiled": arguments.compile,
        "measurements": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
