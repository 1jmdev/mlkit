"""A learned scalar format that works with RTN, GPTQ and incoherence processing."""

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

        def round_columns(values: Tensor, columns: slice) -> mk.Q:
            result = fitted(values, columns)
            result.metadata["trainable"] = ["scales", "values"]
            result.metadata["parameter_formats"] = {"values": "fp16"}
            return result

        return round_columns


def main() -> None:
    weight = torch.randn(128, 256, device="cuda")
    inputs = torch.randn(512, 256, device="cuda")
    format = LearnedCodebook()
    algorithms = {
        "rtn": format,
        "gptq": mk.gptq(format, refit=None),
        "incoherent-gptq": mk.incoherent(mk.gptq(format, refit=None)),
    }
    for name, algorithm in algorithms.items():
        context = mk.Ctx(name=name, X=inputs)
        quantized = algorithm(weight, context)
        bits = quantized.bits + context.additional_bits
        loss = mk.proxy_loss(weight, quantized.w, context)
        print(f"{name:20} {bits / weight.numel():.4f} bpw; proxy loss {float(loss):.6f}")


if __name__ == "__main__":
    main()
