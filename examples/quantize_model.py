"""Quantize a causal model, evaluate it, save it and run compiled generation."""

import argparse
from pathlib import Path

import mlkit as mk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--recipe", default="gptq-int4-g128")
    parser.add_argument("--output", type=Path, default=Path("artifacts/quantized_model"))
    arguments = parser.parse_args()
    model = mk.load(arguments.model, dtype="float16")
    calibration = mk.data("wikitext2", n=8, seq=256)
    evaluation = mk.data("wikitext2", n=16, seq=256, split="test")
    converted = mk.quantize(model, arguments.recipe, calib=calibration)
    print(f"Perplexity: {mk.ppl(converted, data=evaluation, budget='full'):.4f}")
    print(f"Quantized weight precision: {converted.bpw:.3f} bits per weight")
    converted.report()
    converted.save(arguments.output)
    inference = mk.optimize(mk.load(arguments.output), compile=True)
    prompt = model.tokenizer("Explain weight quantization.", return_tensors="pt")
    tokens = inference.generate(**prompt, max_new_tokens=32, do_sample=False)
    print(model.tokenizer.decode(tokens[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
