"""Measure dense and packed scalar-grid inference on representative layer shapes."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

import mlkit as mk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_results/linear_inference.json")
    )
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--device", default="cuda", help="CUDA device identifier")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    dtype = torch.float16
    torch.manual_seed(41)
    records = []
    shapes = [(2048, 2048), (8192, 2048), (2048, 8192), (4096, 4096)]
    for output_width, input_width in shapes:
        module = nn.Sequential(nn.Linear(input_width, output_width, bias=False,
                                         device=device, dtype=dtype))
        converted = mk.quantize(module, mk.int(4, group=128), calib=None)
        packed = mk.optimize(converted, backend="packed")
        for batch in [1, 4, 128]:
            inputs = torch.randn(batch, input_width, device=device, dtype=dtype)
            with torch.inference_mode():
                maximum_error = float((packed(inputs) - converted(inputs)).abs().max())
            for backend, model in [("dense", converted), ("packed", packed)]:
                measurement = mk.benchmark(lambda model=model, inputs=inputs: model(inputs),
                                           device=device,
                                           repetitions=arguments.repetitions, warmup=20)
                record = {
                    "backend": backend, "shape": [output_width, input_width],
                    "batch": batch, "maximum_error": maximum_error,
                    **measurement.to_dict(),
                }
                records.append(record)
                print(json.dumps(record), flush=True)
        del module, converted, packed
    document = {
        "torch": torch.__version__, "device": str(device), "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(device),
        "measurements": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
