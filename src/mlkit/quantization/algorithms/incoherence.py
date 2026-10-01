"""Incoherence processing: orthogonal changes of basis around an inner quantizer."""

from collections.abc import Callable
from typing import Any

from torch import Tensor

from mlkit.quantization.codecs import compose, deterministic_signs, registered
from mlkit.quantization.context import Ctx, layer_seed
from mlkit.quantization.operations.orthogonal_transforms import structured_transform
from mlkit.quantization.protocol import Quantizer, QuantizerFunction
from mlkit.quantization.representation import Q, as_q


class Incoherent(Quantizer):
    def __init__(
        self,
        inner: QuantizerFunction,
        *,
        left: str | None = "rht",
        right: str | None = "rht",
        train_signs: bool = False,
    ) -> None:
        if left not in {None, "rht"} or right not in {None, "rht"}:
            raise ValueError("incoherence transforms must be rht or None")
        self.inner = inner
        self.left = left
        self.right = right
        self.train_signs = train_signs

    def __call__(self, w: Tensor, ctx: Ctx | None = None) -> Q:
        ctx = ctx or Ctx(device=w.device)
        seed = layer_seed(ctx.name, ctx._seed) % 2**32
        left_seed = seed if self.left else None
        right_seed = (seed + 1) % 2**32 if self.right else None
        left_signs = deterministic_signs(w.shape[0], seed, w.device) if self.left else None
        right_signs = deterministic_signs(w.shape[1], seed + 1, w.device) if self.right else None
        transformed = w.float()
        if left_signs is not None:
            transformed = structured_transform((transformed * left_signs[:, None]).T).T
        if right_signs is not None:
            transformed = structured_transform(transformed * right_signs)
        transformed_context = ctx.replace()
        if right_signs is not None:
            original_provider = ctx._provider

            def provider(name: str, fn: Callable | None, reduce: str) -> Tensor:
                if name == "H":
                    hessian = ctx.H.to(w.device) * right_signs[:, None] * right_signs[None, :]
                    return structured_transform(structured_transform(hessian).T).T
                if name == "X":
                    return structured_transform(ctx.X.to(w.device) * right_signs)
                inputs = structured_transform(ctx.X.to(w.device) * right_signs)
                if name == "act_absmean":
                    return inputs.abs().mean(0)
                if name == "act_absmax":
                    return inputs.abs().amax(0)
                if fn is not None:
                    return fn(inputs)
                if original_provider is not None:
                    return original_provider(name, fn, reduce)
                raise KeyError(name)

            transformed_context = ctx.derive(provider)
        quantized = as_q(self.inner(transformed, transformed_context))
        ctx.add_bits(transformed_context.additional_bits)
        if quantized.codes is not None and registered(quantized.codec):
            result = compose(quantized, "basis", {
                "shape": tuple(w.shape),
                "left_seed": left_seed,
                "right_seed": right_seed,
                "left_signs": left_signs if self.train_signs else None,
                "right_signs": right_signs if self.train_signs else None,
            })
            if self.train_signs:
                for name, signs in [("left_signs", left_signs), ("right_signs", right_signs)]:
                    if signs is not None:
                        result.metadata["trainable"].append(name)
                        result.metadata["parameter_formats"][name] = "fp16"
                        ctx.add_bits(16 * signs.numel())
            return result
        if self.train_signs:
            raise ValueError("trainable signs require an inner quantizer with a registered codec")
        reconstruction = quantized.w
        if right_signs is not None:
            reconstruction = structured_transform(reconstruction, inverse=True) * right_signs
        if left_signs is not None:
            reconstruction = (
                structured_transform(reconstruction.T, inverse=True).T * left_signs[:, None]
            )
        return Q(reconstruction, bits=quantized.bits)

    def __repr__(self) -> str:
        return f"incoherent({self.inner!r})"


def incoherent(inner: QuantizerFunction, **options: Any) -> Incoherent:
    return Incoherent(inner, **options)
