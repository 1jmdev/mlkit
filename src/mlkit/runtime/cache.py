"""Online reconstruction of key and value tensors after rotary position embedding."""

from collections.abc import Callable
from typing import Any

from torch import Tensor, nn

from mlkit.quantization.context import Ctx
from mlkit.quantization.formats import Scaled
from mlkit.quantization.representation import as_q


def install_kv_quantization(model: nn.Module, quantizer: Callable) -> Any:
    """Inject a cache adapter so quantization occurs at the post-RoPE update boundary."""
    try:
        from transformers import Cache, DynamicCache
    except ImportError as error:
        raise ImportError("KV quantization requires the transformers extra") from error
    configuration = getattr(model, "config", None)
    if configuration is None or configuration.model_type not in {
        "llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2", "gemma3_text",
    }:
        raise ValueError("KV quantization requires a Llama, Qwen, Mistral or Gemma cache adapter")

    class QuantizedCache(Cache):
        def __init__(self, delegate: Cache) -> None:
            self.delegate = delegate
            self.contexts: dict[tuple[int, str], Ctx] = {}
            self.logical_bits: dict[tuple[int, str], float | None] = {}
            self.quantizer = quantizer

        def __getattr__(self, name: str) -> Any:
            return getattr(object.__getattribute__(self, "delegate"), name)

        @property
        def is_compileable(self) -> bool:
            return False

        def update(
            self, key_states: Tensor, value_states: Tensor, layer_idx: int, *args: Any,
            **kwargs: Any,
        ) -> tuple[Tensor, Tensor]:
            reconstructions = []
            for kind, states in [("key", key_states), ("value", value_states)]:
                identifier = (layer_idx, kind)
                context = self.contexts.setdefault(identifier, Ctx(
                    name=f"cache.layers.{layer_idx}.{kind}", block_idx=layer_idx,
                    device=states.device,
                ))
                flattened = states.reshape(-1, states.shape[-1])
                if isinstance(quantizer, Scaled):
                    reconstruction = quantizer.reconstruct_activations(flattened)
                    reconstructions.append(reconstruction.reshape_as(states))
                    previous = self.logical_bits.get(identifier, 0.0)
                    self.logical_bits[identifier] = None if previous is None else (
                        previous + quantizer.logical_bits((flattened.shape[0], flattened.shape[1]))
                    )
                    continue
                result = as_q(quantizer(flattened.float(), context))
                if result.w.shape != flattened.shape:
                    raise ValueError("KV quantizers must preserve the input tensor shape")
                reconstructions.append(result.w.reshape_as(states).to(states.dtype))
                previous = self.logical_bits.get(identifier, 0.0)
                self.logical_bits[identifier] = (
                    None if previous is None or result.bits is None else previous + result.bits
                )
            return self.delegate.update(
                reconstructions[0], reconstructions[1], layer_idx, *args, **kwargs
            )

    def prepare_cache(
        module: nn.Module, arguments: tuple[Any, ...], keywords: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        cache = keywords.get("past_key_values")
        if cache is None:
            cache = DynamicCache(config=configuration)
        if not isinstance(cache, QuantizedCache):
            cache = QuantizedCache(cache)
        return arguments, keywords | {"past_key_values": cache}

    model.__dict__["_mlkit_kv_quantizer"] = quantizer
    return model.register_forward_pre_hook(prepare_cache, with_kwargs=True)
