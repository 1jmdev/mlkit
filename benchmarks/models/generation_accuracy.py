"""Compare compiled cached logits with eager logits on identical token prefixes."""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

import mlkit as mk


def release_model() -> None:
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_results/generation_accuracy.json")
    )
    arguments = parser.parse_args()
    records = []
    for backend in ("dense", "packed"):
        model = mk.load(arguments.model, dtype="float16")
        if backend == "packed":
            quantized = mk.quantize(model, mk.int(4, group=128), calib=None)
            del model
            model = quantized
            del quantized
        inputs = model.tokenizer(
            "Explain the difference between calibration and inference in weight quantization.",
            return_tensors="pt",
        )
        prompt_length = inputs["input_ids"].shape[1]
        compiled = mk.optimize(model, backend=backend, compile=True, inplace=True)
        with torch.inference_mode():
            result = compiled.generate(
                **inputs,
                do_sample=False,
                min_new_tokens=arguments.tokens,
                max_new_tokens=arguments.tokens,
                pad_token_id=model.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_logits=True,
            )
        generated = result.sequences.cpu()
        cached_logits = torch.stack([value[0].cpu().float() for value in result.logits])
        del result, compiled, model
        release_model()

        reference = mk.load(arguments.model, dtype="float16")
        if backend == "packed":
            quantized = mk.quantize(reference, mk.int(4, group=128), calib=None)
            del reference
            reference = quantized
            del quantized
        with torch.inference_mode():
            output = reference(generated, use_cache=False)
            eager_logits = output.logits[0, prompt_length - 1 : -1].cpu().float()
        targets = generated[0, prompt_length:]
        difference = cached_logits - eager_logits
        records.append({
            "backend": backend,
            "tokens": arguments.tokens,
            "maximum_absolute_logit_error": float(difference.abs().max()),
            "root_mean_square_logit_error": float(difference.square().mean().sqrt()),
            "next_token_agreement": float(
                (cached_logits.argmax(-1) == eager_logits.argmax(-1)).float().mean()
            ),
            "cached_cross_entropy": float(functional.cross_entropy(cached_logits, targets)),
            "eager_cross_entropy": float(functional.cross_entropy(eager_logits, targets)),
        })
        print(json.dumps(records[-1]), flush=True)
        del reference, output
        release_model()
    document = {
        "model": arguments.model,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "float16",
        "gpu": torch.cuda.get_device_name(),
        "measurements": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
