import pytest
import torch

import mlkit as mk
from mlkit.quantization.codecs import decode_feedback, decode_scaled


def tensor_reconstruction(
    codes: torch.Tensor,
    scales: torch.Tensor,
    values: torch.Tensor,
    group: int,
    zero: torch.Tensor | None,
) -> torch.Tensor:
    columns = torch.arange(codes.shape[1]) // group
    reconstruction = values[codes.long()] * scales[:, columns]
    return reconstruction if zero is None else reconstruction + zero[:, columns]


@pytest.mark.parametrize("width", [64, 130, 256])
@pytest.mark.parametrize("group", [8, 32, 64, 96])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_fused_scalar_decode_matches_tensor_operations(
    width: int,
    group: int,
    asymmetric: bool,
) -> None:
    rows = 37
    groups = -(-width // group)
    codes = torch.randint(16, (rows, width), dtype=torch.uint8)
    scales = torch.rand(rows, groups) + 0.1
    zero = torch.randn(rows, groups) if asymmetric else None
    values = torch.linspace(-1, 1, 16)
    expected = tensor_reconstruction(codes, scales, values, group, zero)
    result = decode_scaled(codes, scales=scales, values=values, group=group, zero=zero)
    assert torch.equal(result, expected)
    wide = decode_scaled(codes.int(), scales=scales, values=values, group=group, zero=zero)
    assert torch.equal(wide, expected)


def test_differentiable_decoding_matches_the_fused_kernel() -> None:
    codes = torch.randint(16, (9, 128), dtype=torch.uint8)
    scales = torch.rand(9, 4) + 0.1
    values = torch.linspace(-1, 1, 16)
    fused = decode_scaled(codes, scales=scales, values=values, group=32)
    trainable = scales.clone().requires_grad_(True)
    differentiable = decode_scaled(codes, scales=trainable, values=values, group=32)
    differentiable.sum().backward()
    assert trainable.grad is not None
    assert torch.equal(differentiable.detach(), fused)


def test_group_sizes_without_a_power_of_two_factor_use_tensor_operations() -> None:
    codes = torch.randint(16, (5, 30), dtype=torch.uint8)
    scales = torch.rand(5, 6) + 0.1
    values = torch.linspace(-1, 1, 16)
    result = decode_scaled(codes, scales=scales, values=values, group=5)
    assert torch.equal(result, tensor_reconstruction(codes, scales, values, 5, None))


@pytest.mark.parametrize("refit", [32, 64, 48])
def test_feedback_decoding_matches_error_feedback_output(refit: int) -> None:
    weights = torch.randn(16, 96)
    context = mk.Ctx(X=torch.randn(256, 96))
    result = mk.gptq(mk.int(4, group=16), refit=refit, backend="torch")(weights, context)
    assert result.codec == "feedback"
    reconstruction = decode_feedback(result.codes, **result.params)
    assert torch.equal(reconstruction, result.w)
    columns = torch.arange(96)
    groups_per_region = (refit + 15) // 16
    indices = (columns // refit) * groups_per_region + (columns % refit) // 16
    expected = result.params["values"][result.codes.long()] * result.params["scales"][:, indices]
    assert torch.equal(reconstruction, expected)
