"""Row-aligned grouping and group scale estimation."""

from torch import Tensor


def groups(weight: Tensor, group: int | None) -> Tensor:
    if weight.ndim != 2:
        raise ValueError("groups requires a two-dimensional matrix")
    size = weight.shape[1] if group is None else group
    if size <= 0 or weight.shape[1] % size:
        raise ValueError(
            f"group size {size} must divide the row width {weight.shape[1]}; "
            "use scaled for automatic final-group padding"
        )
    return weight.reshape(-1, size)


def absmax(value: Tensor, qmax: float = 1.0) -> Tensor:
    if qmax <= 0:
        raise ValueError("qmax must be positive")
    return value.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
