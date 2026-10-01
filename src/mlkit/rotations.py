"""Orthogonal structured transforms for ordinary transformer dimensions."""

import functools
import math

import torch
from torch import Tensor

from mlkit.operations import hadamard


def prime(value: int) -> bool:
    return value >= 2 and all(value % divisor for divisor in range(2, math.isqrt(value) + 1))


@functools.lru_cache(maxsize=32)
def paley_matrix(order: int) -> Tensor:
    """Construct real Hadamard matrices using prime-field Paley designs."""
    if prime(order - 1) and (order - 1) % 4 == 3:
        field = order - 1
        indices = torch.arange(field)
        differences = (indices[:, None] - indices[None, :]) % field
        squares = torch.zeros(field)
        squares[(indices[1:] ** 2) % field] = 1
        characters = torch.where(differences == 0, 0.0, 2 * squares[differences] - 1)
        matrix = torch.ones(order, order)
        matrix[1:, 0] = -1
        matrix[1:, 1:] = characters + torch.eye(field)
        return matrix / math.sqrt(order)
    field = order // 2 - 1
    if order % 2 == 0 and prime(field) and field % 4 == 1:
        indices = torch.arange(field)
        differences = (indices[:, None] - indices[None, :]) % field
        squares = torch.zeros(field)
        squares[(indices[1:] ** 2) % field] = 1
        characters = torch.where(differences == 0, 0.0, 2 * squares[differences] - 1)
        conference = torch.ones(field + 1, field + 1)
        conference[0, 0] = 0
        conference[1:, 1:] = characters
        identity = torch.eye(field + 1)
        matrix = torch.cat((
            torch.cat((conference + identity, conference - identity), dim=1),
            torch.cat((conference - identity, -conference - identity), dim=1),
        ), dim=0)
        return matrix / math.sqrt(order)
    raise ValueError(f"no prime-field Paley construction for order {order}")


@functools.lru_cache(maxsize=128)
def transform_factor(width: int) -> tuple[int, int, Tensor | None]:
    if width < 1:
        raise ValueError("transform width must be positive")
    if width & (width - 1) == 0:
        return 1, width, None
    factor = width
    while factor % 2 == 0:
        factor //= 2
    power = width // factor
    for multiplier in [1, 2, 4, 8, 16]:
        order = factor * multiplier
        if power % multiplier:
            continue
        try:
            return order, power // multiplier, paley_matrix(order)
        except ValueError:
            continue
    # An odd dimension cannot have a real Hadamard matrix. An orthonormal DCT
    # factor preserves exact inversion and avoids padding or a dense full basis.
    indices = torch.arange(factor, dtype=torch.float64)
    matrix = torch.cos(math.pi / factor * (indices[None, :] + 0.5) * indices[:, None])
    matrix[0] *= 1 / math.sqrt(factor)
    matrix[1:] *= math.sqrt(2 / factor)
    return factor, power, matrix.float()


def structured_transform(value: Tensor, *, inverse: bool = False) -> Tensor:
    factor, power, basis = transform_factor(value.shape[-1])
    if basis is None:
        return hadamard(value)
    grouped = value.reshape(*value.shape[:-1], factor, power)
    basis = basis.to(device=value.device, dtype=value.dtype)
    transformed = hadamard(grouped).transpose(-1, -2)
    transformed = transformed @ (basis if inverse else basis.T)
    return transformed.transpose(-1, -2).reshape_as(value)


def randomized_transform(value: Tensor, *, seed: int = 0, inverse: bool = False) -> Tensor:
    generator = torch.Generator(device=value.device).manual_seed(seed)
    signs = torch.randint(2, (value.shape[-1],), generator=generator, device=value.device)
    signs = signs.to(value.dtype) * 2 - 1
    return structured_transform(value, inverse=True) * signs if inverse else structured_transform(
        value * signs
    )
