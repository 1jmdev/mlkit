"""Compare scalar, lattice and trellis formats through the same error-feedback API."""

import torch

import mlkit as mk


def main() -> None:
    weight = torch.randn(32, 64, device="cuda")
    inputs = torch.randn(256, 64, device="cuda")
    methods = {
        "scalar-int2": mk.gptq(mk.int(2, group=32)),
        "lattice-e8p": mk.incoherent(mk.ldlq(mk.scaled(mk.grid.e8p(), group=None), step=8)),
        "trellis-l8": mk.incoherent(mk.ldlq(mk.trellis(L=8, tile=16), step=16)),
    }
    for name, quantization in methods.items():
        context = mk.Ctx(name=name, X=inputs)
        result = quantization(weight, context)
        bits = result.bits + context.additional_bits
        loss = mk.proxy_loss(weight, result.w, context)
        print(f"{name:16} {bits / weight.numel():.4f} bpw; proxy loss {float(loss):.6f}")


if __name__ == "__main__":
    main()
