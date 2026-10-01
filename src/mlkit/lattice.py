"""E8P lattice geometry with exact sign-parity nearest-neighbor search.

The point set follows the mathematical E8P definition in QuIP# (ICML 2024).
Search uses absolute magnitudes and an explicit minimum-cost parity correction,
without enumerating all sign combinations for each input vector.
"""

import functools

import torch
from torch import Tensor

NORM_TWELVE_MASKS = (
    241, 242, 244, 248, 55, 87, 103, 151, 167, 199,
    59, 91, 107, 155, 171, 203, 61, 93, 109, 157, 173,
    206, 62, 94, 110, 158, 174, 236, 115,
)


@functools.lru_cache(maxsize=1)
def absolute_points() -> Tensor:
    coordinates = torch.tensor([0.5, 1.5, 2.5])
    candidates = torch.cartesian_prod(*(coordinates for _ in range(8)))
    interior = candidates[candidates.square().sum(1) <= 10]
    masks = torch.tensor(NORM_TWELVE_MASKS)
    boundary = 0.5 + ((masks[:, None] >> torch.arange(8)) & 1).float()
    return torch.cat((interior, boundary))


@functools.lru_cache(maxsize=1)
def e8p_points() -> Tensor:
    magnitudes = absolute_points()
    first_signs = (torch.arange(128)[:, None] >> torch.arange(7)) & 1
    last_sign = (first_signs.sum(1)[None, :] + magnitudes.sum(1)[:, None].long()) % 2
    signs = torch.cat((first_signs[None].expand(256, -1, -1), last_sign[..., None]), dim=2)
    lattice = (magnitudes[:, None, :] * (1 - 2 * signs)).reshape(-1, 8)
    return torch.cat((lattice + 0.25, lattice - 0.25))


def nearest_e8p(value: Tensor, *, chunk: int = 512) -> Tensor:
    if value.shape[-1] != 8 or chunk < 1:
        raise ValueError(
            "E8P quantization requires eight-dimensional vectors and positive chunk size"
        )
    magnitudes = absolute_points().to(device=value.device)
    magnitude_norms = magnitudes.square().sum(1)
    magnitude_parities = magnitudes.sum(1).long() % 2
    outputs = []
    for samples in value.reshape(-1, 8).float().split(chunk):
        selected = torch.empty_like(samples)
        minimum_cost = torch.full((len(samples),), float("inf"), device=value.device)
        for shift in [-0.25, 0.25]:
            centered = samples - shift
            absolute = centered.abs()
            signs = torch.where(centered < 0, -1.0, 1.0)
            sign_parities = (centered < 0).sum(1) % 2
            mismatch = sign_parities[:, None] != magnitude_parities[None, :]
            products = absolute[:, None, :] * magnitudes[None, :, :]
            correction_cost, correction_axis = products.min(2)
            costs = (
                absolute.square().sum(1)[:, None] + magnitude_norms[None, :]
                - 2 * absolute @ magnitudes.T + 4 * correction_cost * mismatch
            )
            best_cost, indices = costs.min(1)
            rows = torch.arange(len(samples), device=value.device)
            correction = correction_axis[rows, indices]
            corrected_signs = signs.clone()
            corrected_signs[rows, correction] *= torch.where(
                mismatch[rows, indices], -1.0, 1.0
            )
            reconstructed = magnitudes[indices] * corrected_signs + shift
            improved = best_cost < minimum_cost
            selected = torch.where(improved[:, None], reconstructed, selected)
            minimum_cost = torch.minimum(minimum_cost, best_cost)
        outputs.append(selected)
    return torch.cat(outputs).reshape_as(value).to(value.dtype) if outputs else value.clone()
