"""Compare format reconstruction on a selected model layer before full conversion."""

import argparse
from pathlib import Path

import mlkit as mk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--layers", default="model.layers.0.self_attn.q_proj")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    model = mk.load(arguments.model, dtype="float16")
    calibration = mk.data("wikitext2", n=4, seq=128, seed=17)
    methods = [
        mk.gptq(mk.int(2, group=64)),
        mk.incoherent(mk.ldlq(mk.scaled(mk.grid.e8p(), group=None), step=8, refit=None)),
        mk.incoherent(mk.ldlq(mk.trellis(L=8), step=16, refit=None)),
    ]
    table = mk.probe(methods, model, layers=arguments.layers, calib=calibration)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(arguments.output)


if __name__ == "__main__":
    main()
