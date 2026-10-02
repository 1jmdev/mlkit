"""Portable unsigned bit streams with little-endian bit ordering.

Precisions that divide a byte, and sixteen bits, take direct paths without wide
temporaries. Other precisions assemble each byte from the codes that overlap it.
"""

import math

import torch
from torch import Tensor


def pack(codes: Tensor, bits: int) -> Tensor:
    """Pack unsigned codes into bytes, including non-byte-aligned precisions."""
    if not 1 <= bits <= 16:
        raise ValueError("packing supports one through sixteen bits")
    flattened = codes.detach().flatten()
    if flattened.is_floating_point():
        raise TypeError("codes must have an integer dtype")
    if flattened.numel() and (int(flattened.min()) < 0 or int(flattened.max()) >= 2**bits):
        raise ValueError("codes exceed the declared unsigned bit capacity")
    if bits == 8:
        return flattened.to(torch.uint8)
    if bits == 16:
        wide = flattened.to(torch.int32)
        return torch.stack((wide & 255, wide >> 8), dim=1).to(torch.uint8).flatten()
    if 8 % bits == 0:
        return pack_within_bytes(flattened, bits)
    return pack_across_bytes(flattened, bits)


def pack_within_bytes(flattened: Tensor, bits: int) -> Tensor:
    """Pack precisions of one, two or four bits, where no code crosses a byte."""
    codes_per_byte = 8 // bits
    padding = (-flattened.numel()) % codes_per_byte
    narrow = flattened.to(torch.uint8)
    if padding:
        narrow = torch.cat((narrow, narrow.new_zeros(padding)))
    grouped = narrow.reshape(-1, codes_per_byte)
    packed = grouped[:, 0].clone()
    for position in range(1, codes_per_byte):
        packed |= grouped[:, position] << (position * bits)
    return packed


def pack_across_bytes(flattened: Tensor, bits: int) -> Tensor:
    """Pack precisions whose codes cross byte boundaries, one periodic group at a time.

    Below eight bits the arithmetic stays in bytes, where a left shift discards
    the bits that belong to the next byte.
    """
    count = flattened.numel()
    bytes_per_group = bits // math.gcd(bits, 8)
    codes_per_group = 8 // math.gcd(bits, 8)
    padding = (-count) % codes_per_group
    if padding:
        flattened = torch.cat((flattened, flattened.new_zeros(padding)))
    narrow = bits < 8
    grouped = flattened.reshape(-1, codes_per_group).to(torch.uint8 if narrow else torch.int32)
    output = torch.empty(
        (len(grouped), bytes_per_group), dtype=torch.uint8, device=flattened.device
    )
    for byte in range(bytes_per_group):
        assembled = None
        for column in range(codes_per_group):
            shift = column * bits - byte * 8
            if shift >= 8 or shift <= -bits:
                continue
            contribution = (
                grouped[:, column] << shift if shift >= 0 else grouped[:, column] >> -shift
            )
            if not narrow:
                contribution &= 255
            assembled = contribution if assembled is None else assembled.bitwise_or_(contribution)
        assert assembled is not None
        output[:, byte] = assembled
    return output.flatten()[: math.ceil(count * bits / 8)]


def unpack(packed: Tensor, bits: int, shape: tuple[int, ...]) -> Tensor:
    """Decode an unsigned bit stream without platform-dependent integer views."""
    if not 1 <= bits <= 16 or packed.dtype != torch.uint8:
        raise ValueError("unpack requires uint8 storage and one through sixteen bits")
    count = math.prod(shape)
    if count < 0 or any(dimension < 0 for dimension in shape):
        raise ValueError("shape dimensions must be nonnegative")
    if packed.numel() != math.ceil(count * bits / 8):
        raise ValueError("packed byte count does not match the declared shape and precision")
    stream = packed.flatten()
    if bits == 8:
        return stream.reshape(shape).clone()
    if bits == 16:
        pairs = stream.reshape(-1, 2).to(torch.int32)
        return (pairs[:, 0] | (pairs[:, 1] << 8)).reshape(shape)
    if 8 % bits == 0:
        mask = 2**bits - 1
        fields = [(stream >> shift) & mask for shift in range(0, 8, bits)]
        return torch.stack(fields, dim=1).flatten()[:count].reshape(shape)
    return unpack_across_bytes(stream, bits, count).reshape(shape)


def unpack_across_bytes(stream: Tensor, bits: int, count: int) -> Tensor:
    """Decode precisions whose codes cross byte boundaries, one periodic group at a time.

    A code of fewer than eight bits spans at most two bytes of its group and is
    assembled in byte arithmetic. A wider code is read through a three-byte
    window; two zero guard bytes per group keep the window in range.
    """
    bytes_per_group = bits // math.gcd(bits, 8)
    codes_per_group = 8 // math.gcd(bits, 8)
    padding = (-stream.numel()) % bytes_per_group
    if padding:
        stream = torch.cat((stream, stream.new_zeros(padding)))
    groups = stream.reshape(-1, bytes_per_group)
    mask = 2**bits - 1
    if bits < 8:
        output = torch.empty(
            (len(groups), codes_per_group), dtype=torch.uint8, device=stream.device
        )
        for column in range(codes_per_group):
            byte, shift = divmod(column * bits, 8)
            code = groups[:, byte] >> shift
            if shift + bits > 8:
                code.bitwise_or_(groups[:, byte + 1] << (8 - shift))
            output[:, column] = code.bitwise_and_(mask)
        return output.flatten()[:count]
    grouped = torch.cat((groups, groups.new_zeros(len(groups), 2)), dim=1).to(torch.int32)
    output = torch.empty(
        (len(grouped), codes_per_group), dtype=torch.int32, device=stream.device
    )
    for column in range(codes_per_group):
        byte, shift = divmod(column * bits, 8)
        window = grouped[:, byte] | (grouped[:, byte + 1] << 8) | (grouped[:, byte + 2] << 16)
        output[:, column] = (window >> shift) & mask
    return output.flatten()[:count]
