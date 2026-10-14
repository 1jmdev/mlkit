import argparse
import json
from pathlib import Path

import torch

import mlkit as mk

E2M1 = torch.tensor([-6, -4, -3, -2, -1.5, -1, -0.5, -0.0, 0.0, 0.5, 1, 1.5, 2, 3, 4, 6])


@mk.codec("example_nvfp4", row_parameters=("scales",))
def decode_nvfp4(codes, *, scales, tensor_scale):
    return (
        E2M1.to(codes.device)[codes.long()].reshape(*scales.shape, 16)
        * scales[..., None]
        * tensor_scale
    ).reshape_as(codes)


@mk.quantizer
def nvfp4_gptq(weight, context):
    tensor_scale = weight.float().abs().amax().clamp_min(1e-12) / (6 * 448)
    result = mk.gptq(
        mk.scaled(
            mk.grid.values(E2M1.to(weight.device) * tensor_scale, bits=4),
            group=16,
            scale="mse",
            scale_fmt="fp8",
        ),
        refit=None,
    )(weight, context)
    return mk.Q(
        codes=result.codes,
        params={"scales": result.params["scales"], "tensor_scale": tensor_scale},
        decode=decode_nvfp4,
        codec="example_nvfp4",
        bits=4 * weight.numel() + 8 * result.params["scales"].numel() + 32,
        metadata={"code_bits": 4, "parameter_formats": {"scales": "fp8"}},
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output", type=Path, default=Path("artifacts/llama_nvfp4_dbs"))
    parser.add_argument("--calibration-sequences", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--stability-samples", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--decision-sequences", type=int, default=4)
    parser.add_argument("--decision-length", type=int, default=256)
    arguments = parser.parse_args()
    source = mk.load(arguments.model, dtype="float16")
    decisions = mk.data(
        "wikitext2",
        n=arguments.decision_sequences,
        seq=arguments.decision_length,
        split="test",
        tokenizer=source.tokenizer,
    )
    print("Recording FP16 next-token decisions on the held-out probe.", flush=True)
    original_decisions = [
        source(**batch, use_cache=False).logits[:, :-1].argmax(-1).cpu()
        for batch in decisions
    ]
    converted = mk.quantize(
        source,
        nvfp4_gptq,
        calib=mk.data(
            "wikitext2",
            n=arguments.calibration_sequences,
            seq=arguments.sequence_length,
            seed=0,
        ),
        sample_rows=arguments.stability_samples,
        calibration_storage="host",
        cache_dir=None,
        stability=True,
        stability_top_k=arguments.top_k,
        progress=True,
    )
    source.module.cpu()
    squared_error = torch.zeros((), dtype=torch.float64)
    elements = 0
    for name in converted.quantized:
        original = source.module.get_submodule(name).weight
        reconstruction = converted.module.get_submodule(name).weight.cpu()
        for original_rows, reconstructed_rows in zip(
            original.split(128), reconstruction.split(128)
        ):
            squared_error += (
                original_rows.float() - reconstructed_rows.float()
            ).square().double().sum()
        elements += original.numel()
    tested = sum(
        record["tested_local_decisions"] for record in converted.stability_reports.values()
    )
    metrics = {
        "reference_model": arguments.model,
        "nvfp4_mse_original": float(squared_error / elements),
        "local_channel_dbs": sum(
            round(record["local_channel_dbs"] * record["tested_local_decisions"])
            for record in converted.stability_reports.values()
        ) / tested,
        "tested_local_decisions": tested,
        "top_k": arguments.top_k,
        "layers": converted.stability_reports,
    }
    print(f"\nSaving checkpoint: {arguments.output}", flush=True)
    converted.save(arguments.output, overwrite=True)
    del source, converted
    torch.cuda.empty_cache()
    restored = mk.load(arguments.output)
    unchanged = 0
    tested = 0
    for index, (batch, original) in enumerate(zip(decisions, original_decisions)):
        reconstructed = restored(**batch, use_cache=False).logits[:, :-1].argmax(-1).cpu()
        unchanged += int((original == reconstructed).sum())
        tested += original.numel()
        print(
            f"Next-token probe {index + 1}/{len(decisions)}  |  "
            f"DBS {unchanged / tested:7.2%}  |  {tested:,} decisions",
            flush=True,
        )
    metrics["next_token_dbs"] = unchanged / tested
    metrics["tested_next_token_decisions"] = tested
    print("Evaluating full WikiText-2 perplexity.", flush=True)
    metrics["nvfp4_ppl_wikitext2"] = mk.ppl(
        restored, data="wikitext2", budget="full", seq=arguments.sequence_length
    )
    metrics["quantized_bpw"] = restored.bpw
    (arguments.output / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"\n{'Next-token DBS':<26} {metrics['next_token_dbs']:.2%}\n"
        f"{'Local channel DBS':<26} {metrics['local_channel_dbs']:.2%}\n"
        f"{'Weight MSE':<26} {metrics['nvfp4_mse_original']:.8g}\n"
        f"{'WikiText-2 perplexity':<26} {metrics['nvfp4_ppl_wikitext2']:.4f}\n"
        f"{'Quantized bits/weight':<26} {metrics['quantized_bpw']:.4f}\n"
        f"{'Metrics':<26} {arguments.output / 'evaluation.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
