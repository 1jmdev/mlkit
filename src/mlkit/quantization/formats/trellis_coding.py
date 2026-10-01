"""Bounded-memory free-start trellis quantization of weight tiles."""

from collections.abc import Callable

import torch
from torch import Tensor

from mlkit.quantization.codecs.trellis import decode_trellis
from mlkit.quantization.context import Ctx
from mlkit.quantization.operations.trellis_search import one_mad, viterbi
from mlkit.quantization.protocol import Quantizer
from mlkit.quantization.representation import Q


class Trellis(Quantizer):
    def __init__(
        self,
        L: int = 12,
        k: int = 2,
        tile: int = 16,
        chunk: int = 64,
        code: Callable[[Tensor], Tensor] = one_mad,
        *,
        backend: str = "auto",
        memory_budget: int = 64 * 1024 * 1024,
    ) -> None:
        if not 1 <= k <= min(8, L) or not 1 <= L <= 16 or tile < 1 or chunk < 1:
            raise ValueError("invalid trellis precision, state size, tile, or chunk")
        if memory_budget < (tile * tile - 1) * 2 ** (L - k):
            raise ValueError("trellis memory budget cannot hold even one sequence traceback")
        self.L = L
        self.k = k
        self.tile = tile
        self.chunk = chunk
        self.code = code
        self.backend = backend
        self.memory_budget = memory_budget

    def fit(self, w: Tensor, ctx: Ctx) -> Callable[[Tensor, slice], Q]:
        if w.shape[0] % self.tile or w.shape[1] % self.tile:
            raise ValueError("trellis matrix dimensions must be multiples of the tile width")
        scale = w.float().square().mean().sqrt().clamp(2**-24, 65504).half().float()
        ctx.add_bits(16)
        codebook = self.code(torch.arange(2**self.L)).float()
        custom_codebook = None if self.code is one_mad else codebook.half().float()
        if custom_codebook is not None:
            ctx.add_bits(16 * custom_codebook.numel())
            codebook = custom_codebook
        codebook = codebook.to(w.device)
        length = self.tile * self.tile
        traceback_bytes = max(1, (length - 1) * 2 ** (self.L - self.k))
        chunk = min(self.chunk, max(1, self.memory_budget // traceback_bytes))

        def round_columns(value: Tensor, columns: slice) -> Q:
            if value.shape[1] != self.tile:
                if value.shape[1] % self.tile:
                    raise ValueError("trellis rounder column width must be a multiple of tile")
                pieces = []
                for start in range(0, value.shape[1], self.tile):
                    pieces.append(round_columns(value[:, start : start + self.tile], slice(None)))
                parameters = pieces[0].params | {
                    "shape": tuple(value.shape),
                    "initial_states": torch.cat([
                        piece.params["initial_states"] for piece in pieces
                    ]),
                }
                return Q(
                    codes=torch.cat([piece.codes for piece in pieces if piece.codes is not None]),
                    params=parameters,
                    decode=decode_trellis,
                    codec="trellis",
                    metadata=pieces[0].metadata,
                    bits=sum(piece.bits for piece in pieces if piece.bits is not None),
                )
            sequences = (value.float() / scale).reshape(-1, length)
            states = torch.cat([
                viterbi(part, codebook, self.L, self.k, backend=self.backend, return_states=True)
                for part in sequences.split(chunk)
            ])
            transitions = (states[:, 1:] & (2**self.k - 1)).to(torch.uint8)
            initial_states = states[:, 0]
            bits = self.k * transitions.numel() + self.L * initial_states.numel()
            return Q(
                codes=transitions,
                params={
                    "initial_states": initial_states,
                    "scale": scale,
                    "L": self.L,
                    "k": self.k,
                    "shape": tuple(value.shape),
                    "tile": self.tile,
                    "codebook": custom_codebook,
                },
                decode=decode_trellis,
                codec="trellis",
                bits=bits,
                metadata={
                    "code_bits": self.k,
                    "trainable": ["scale"],
                    "parameter_bits": {"initial_states": self.L},
                    "parameter_formats": {"scale": "fp16", "codebook": "fp16"},
                },
            )

        return round_columns

    def __repr__(self) -> str:
        return f"trellis(L={self.L}, k={self.k}, tile={self.tile})"


def trellis(**options) -> Trellis:
    return Trellis(**options)
