"""Portable unsigned bit streams with little-endian bit ordering."""

import math

import torch
from torch import Tensor


def pack(codes: Tensor, bits: int) -> Tensor:
    """Pack unsigned codes into bytes, including non-byte-aligned precisions."""
    if not 1 <= bits <= 16:
        raise ValueError("packing supports one through sixteen bits")
    flattened = codes.detach().flatten()
    if flattened.numel() and ((flattened < 0).any() or (flattened >= 2**bits).any()):
        raise ValueError("codes exceed the declared unsigned bit capacity")
    if flattened.is_floating_point():
        raise TypeError("codes must have an integer dtype")
    bytes_per_group = bits // math.gcd(bits, 8)
    codes_per_group = 8 // math.gcd(bits, 8)
    padding = (-len(flattened)) % codes_per_group
    if padding:
        flattened = torch.cat((flattened, flattened.new_zeros(padding)))
    grouped = flattened.reshape(-1, codes_per_group).to(torch.int32)
    output = torch.zeros(
        (len(grouped), bytes_per_group), dtype=torch.uint8, device=codes.device
    )
    for byte in range(bytes_per_group):
        assembled = torch.zeros(len(grouped), dtype=torch.int32, device=codes.device)
        for column in range(codes_per_group):
            shift = column * bits - byte * 8
            if shift >= 8 or shift <= -bits:
                continue
            contribution = (
                grouped[:, column] << shift if shift >= 0 else grouped[:, column] >> -shift
            )
            assembled |= contribution & 255
        output[:, byte] = assembled.to(torch.uint8)
    return output.flatten()[: math.ceil(codes.numel() * bits / 8)]


def unpack(packed: Tensor, bits: int, shape: tuple[int, ...]) -> Tensor:
    """Decode an unsigned bit stream without platform-dependent integer views."""
    if not 1 <= bits <= 16 or packed.dtype != torch.uint8:
        raise ValueError("unpack requires uint8 storage and one through sixteen bits")
    count = math.prod(shape)
    if count < 0 or any(dimension < 0 for dimension in shape):
        raise ValueError("shape dimensions must be nonnegative")
    if packed.numel() != math.ceil(count * bits / 8):
        raise ValueError("packed byte count does not match the declared shape and precision")
    positions = torch.arange(count, device=packed.device, dtype=torch.long) * bits
    offsets, shifts = positions // 8, positions % 8
    storage = torch.cat((packed.flatten(), packed.new_zeros(2))).to(torch.int32)
    values = storage[offsets] | (storage[offsets + 1] << 8) | (storage[offsets + 2] << 16)
    codes = (values >> shifts) & (2**bits - 1)
    return codes.reshape(shape).to(torch.uint8 if bits <= 8 else torch.int32)
