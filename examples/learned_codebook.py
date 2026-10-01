"""A learned scalar format that works with RTN, GPTQ and incoherence processing."""

import argparse
from collections.abc import Callable

import torch
from torch import Tensor

import mlkit as mk


class LearnedCodebook(mk.Quantizer):
    def __init__(self, bits: int = 4, group: int = 64) -> None:
        self.bits = bits
        self.group = group

    def fit(self, weight: Tensor, context: mk.Ctx) -> Callable:
        grouped = mk.groups(weight, self.group)
        scales = mk.absmax(grouped).half().float()
        normalized = (grouped / scales).flatten()
        importance = context.H.diagonal().expand_as(weight).flatten()
        codebook = mk.kmeans(normalized, k=2**self.bits, weights=importance,
                             iters=10, seed=context.rng.initial_seed()).half().float()
        context.add_bits(16 * codebook.numel())
        format = mk.scaled(mk.grid.values(codebook, bits=self.bits), group=self.group,
                           scale=lambda values: mk.absmax(values))
        fitted = format.fit(weight, context)

        return fitted.with_metadata(
            trainable=["scales", "values"], parameter_formats={"values": "fp16"}
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    arguments = parser.parse_args()
    format = LearnedCodebook()
    algorithms = {
        "rtn": format,
        "gptq": mk.gptq(format, refit=None),
        "incoherent-gptq": mk.incoherent(mk.gptq(format, refit=None)),
    }
    if arguments.model is not None:
        model = mk.load(arguments.model, dtype="float16")
        calibration = mk.data("wikitext2", n=4, seq=128, seed=17)
        evaluation = mk.data("wikitext2", n=4, seq=128, split="test")
        recipes = [mk.Recipe(weights=mk.int(4, group=64), name="uniform-int4")]
        recipes.extend(
            mk.Recipe(weights=algorithm, name=name) for name, algorithm in algorithms.items()
        )
        mk.compare(model, recipes, calib=calibration, data=evaluation, budget="full", seq=128)
        return
    weight = torch.randn(128, 256, device="cuda")
    inputs = torch.randn(512, 256, device="cuda")
    for name, algorithm in algorithms.items():
        context = mk.Ctx(name=name, X=inputs)
        quantized = algorithm(weight, context)
        bits = quantized.bits + context.additional_bits
        loss = mk.proxy_loss(weight, quantized.w, context)
        print(f"{name:20} {bits / weight.numel():.4f} bpw; proxy loss {float(loss):.6f}")


if __name__ == "__main__":
    main()
