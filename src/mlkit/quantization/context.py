"""Layer context and demand-driven calibration statistics."""

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn


def layer_seed(name: str, seed: int = 0) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def copy_cache(cache: dict[str, Any]) -> dict[str, Any]:
    """Copy mutable bookkeeping containers while sharing tensor and module storage."""
    def copy_container(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: copy_container(item) for key, item in value.items()}
        if isinstance(value, list):
            return [copy_container(item) for item in value]
        if isinstance(value, tuple):
            return tuple(copy_container(item) for item in value)
        if isinstance(value, set):
            return value.copy()
        return value

    return copy_container(cache)


class Ctx:
    """Statistics are supplied only when requested by the quantizer.

    Standalone quantizers can use ``Ctx(H=...)`` or ``Ctx(X=...)``. The model
    engine supplies a provider that replays calibration when a statistic is read.
    """

    def __init__(
        self,
        name: str = "",
        module: nn.Module | None = None,
        block: nn.Module | None = None,
        block_idx: int = 0,
        *,
        H: Tensor | None = None,
        X: Tensor | None = None,
        stats: Mapping[str, Tensor] | None = None,
        provider: Callable[[str, Callable[[Tensor], Tensor] | None, str], Tensor] | None = None,
        cache: dict[str, Any] | None = None,
        siblings: tuple[str, ...] = (),
        seed: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        self.name = name
        self.module = module
        self.block = block
        self.block_idx = block_idx
        self.siblings = siblings
        self.cache = cache if cache is not None else {}
        self._stats = dict(stats or {})
        if H is not None:
            self._stats["H"] = H
        if X is not None:
            self._stats["X"] = X
        self._provider = provider
        self._seed = seed
        statistic_device = H.device if H is not None else X.device if X is not None else "cpu"
        self._device = torch.device(statistic_device if device is None else device)
        self.rng = torch.Generator(device=self._device).manual_seed(layer_seed(name, seed))
        self._additional_bits = 0.0

    def stat(
        self,
        name: str,
        fn: Callable[[Tensor], Tensor] | None = None,
        reduce: str = "mean",
    ) -> Tensor:
        if reduce not in {"mean", "max", "sum", "sample"}:
            raise ValueError("stat reduction must be mean, max, sum, or sample")
        if name not in self._stats:
            if self._provider is not None:
                self._stats[name] = self._provider(name, fn, reduce)
            elif "X" in self._stats:
                inputs = self._stats["X"].float()
                if name == "H":
                    self._stats[name] = inputs.T @ inputs / inputs.shape[0]
                elif name == "act_absmean":
                    self._stats[name] = inputs.abs().mean(0)
                elif name == "act_absmax":
                    self._stats[name] = inputs.abs().amax(0)
                elif fn is not None:
                    self._stats[name] = fn(inputs)
                else:
                    raise KeyError(f"no statistic implementation for {name!r}")
            else:
                raise RuntimeError(
                    f"layer {self.name!r} needs calibration statistic {name!r}; "
                    "supply calibration data to quantize, or construct Ctx(X=...) / Ctx(H=...)"
                )
        return self._stats[name].to(self._device)

    @property
    def H(self) -> Tensor:
        return self.stat("H")

    @property
    def X(self) -> Tensor:
        return self.stat("X", reduce="sample")

    @property
    def act_absmean(self) -> Tensor:
        return self.stat("act_absmean")

    @property
    def act_absmax(self) -> Tensor:
        return self.stat("act_absmax", reduce="max")

    def add_bits(self, bits: int | float) -> None:
        if bits < 0:
            raise ValueError("additional bits must be nonnegative")
        self._additional_bits += bits

    @property
    def additional_bits(self) -> float:
        """Side-information bits declared through add_bits during this quantizer call."""
        return self._additional_bits

    def derive(
        self,
        provider: Callable[[str, Callable[[Tensor], Tensor] | None, str], Tensor],
        **overrides: Any,
    ) -> "Ctx":
        """A context for transformed weights whose statistics come from ``provider``.

        Algorithms that permute, rotate or rescale the weight columns use this to
        supply the inner quantizer with statistics expressed in the transformed
        basis. Statistics are computed only when the inner quantizer reads them.
        """
        derived = self.replace(**overrides)
        derived._stats = {}
        derived._provider = provider
        return derived

    def replace(self, **overrides: Any) -> "Ctx":
        copied = Ctx(
            self.name,
            self.module,
            self.block,
            self.block_idx,
            stats=self._stats,
            provider=self._provider,
            cache=self.cache,
            siblings=self.siblings,
            seed=self._seed,
            device=self._device,
        )
        for name, value in overrides.items():
            if name in {"H", "X", "act_absmean", "act_absmax"}:
                copied._stats[name] = value
            else:
                setattr(copied, name, value)
        return copied
