"""Per-layer and per-pass records produced during model conversion."""

from dataclasses import dataclass


@dataclass
class LayerReport:
    name: str
    shape: tuple[int, int]
    bits: float | None
    elements: int
    loss: float
    seconds: float
    method: str

    @property
    def bpw(self) -> float | None:
        return None if self.bits is None else self.bits / self.elements


@dataclass
class BlockPassReport:
    name: str
    block_idx: int
    seconds: float
    trainable_elements: int
    steps: int | None
    initial_loss: float | None
    final_loss: float | None
