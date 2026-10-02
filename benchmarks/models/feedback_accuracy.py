"""Compare fused feedback against the PyTorch reference on calibrated model layers."""

import argparse
import fnmatch
import json
import time
from pathlib import Path

import torch
from torch import nn

import mlkit as mk
from mlkit.calibration import BlockStatistics, CalibrationSession


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--layers", default="*")
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_results/feedback_accuracy.json")
    )
    arguments = parser.parse_args()
    model = mk.load(arguments.model, dtype="float16")
    calibration = mk.data(
        "wikitext2", n=arguments.calibration_batches, seq=arguments.sequence,
        tokenizer=model.tokenizer, seed=17,
    )
    session = CalibrationSession(
        model, calibration, sequential=False, sample_rows=4096,
        cache_dir=None, need_targets=False, selected_blocks=(arguments.block,),
    )
    block = model.blocks[arguments.block]
    prefix = model.architecture.block_name(arguments.block)
    statistics = BlockStatistics(session, block, arguments.block, prefix)
    measurements = []
    with torch.no_grad():
        for relative_name, module in block.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            name = f"{prefix}.{relative_name}"
            if not fnmatch.fnmatchcase(name, arguments.layers):
                continue
            weight = module.weight.float()
            context = mk.Ctx(
                name, module, block, arguments.block, device=weight.device,
                provider=statistics.provider(name, weight.device),
            )
            hessian = context.H
            format = mk.int(4, group=128)
            reconstructed = {}
            record = {"layer": name, "shape": list(weight.shape)}
            for method, quantizer in (
                ("rtn", format),
                ("torch", mk.gptq(format, backend="torch")),
                ("triton", mk.gptq(format, backend="triton")),
            ):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = quantizer(weight, mk.Ctx(H=hessian))
                reconstruction = result.w
                loss = mk.proxy_loss(weight, reconstruction, mk.Ctx(H=hessian)).item()
                torch.cuda.synchronize()
                record[method] = {"loss": loss, "seconds": time.perf_counter() - start}
                if method != "rtn":
                    reconstructed[method] = reconstruction
            difference = reconstructed["torch"] - reconstructed["triton"]
            record["maximum_error"] = difference.abs().max().item()
            record["different_fraction"] = (difference != 0).float().mean().item()
            measurements.append(record)
            print(json.dumps(record), flush=True)
    document = {
        "model": arguments.model, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "sequence": arguments.sequence, "seed": 17,
        "calibration_batches": arguments.calibration_batches, "measurements": measurements,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
