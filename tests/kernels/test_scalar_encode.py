import pytest
import torch

import mlkit as mk
from mlkit.kernels import scalar_encode

FORMATS = {
    "int4": lambda **options: mk.int(4, **options),
    "int3": lambda **options: mk.int(3, **options),
    "int8": lambda **options: mk.int(8, **options),
    "nf4": lambda **options: mk.nf4(**options),
    "e2m1": lambda **options: mk.scaled(mk.grid.fp("e2m1"), **options),
    "e4m3": lambda **options: mk.scaled(mk.grid.fp("e4m3"), **options),
}


def reference(format, weights: torch.Tensor, monkeypatch) -> mk.Q:
    """Round with tensor operations by withholding the kernel tile."""
    with monkeypatch.context() as patch:
        patch.setattr(scalar_encode, "tile_for", lambda group: None)
        return format(weights)


@pytest.mark.parametrize("name", FORMATS)
@pytest.mark.parametrize("group", [16, 64, None])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_fused_encoding_matches_tensor_reference(
    monkeypatch,
    name: str,
    group: int | None,
    asymmetric: bool,
) -> None:
    weights = torch.randn(37, 200) * (torch.rand(37, 1) * 3 + 0.1)
    format = FORMATS[name](group=group, asym=asymmetric)
    expected = reference(format, weights, monkeypatch)
    result = format(weights)
    assert result.codes.dtype == expected.codes.dtype == torch.uint8
    assert torch.equal(result.params["scales"], expected.params["scales"])
    assert torch.equal(result.codes, expected.codes)
    assert torch.equal(result.w, expected.w)
    assert result.bits == expected.bits


def test_fused_encoding_is_exact_on_rounding_ties(monkeypatch) -> None:
    weights = torch.randn(64, 4096).half().float()
    for format in (mk.int(4, group=128), mk.nf4(group=64), mk.int(4, group=128, asym=True)):
        expected = reference(format, weights, monkeypatch)
        result = format(weights)
        assert torch.equal(result.codes, expected.codes)


@pytest.mark.parametrize("name", ["int4", "nf4", "e2m1"])
@pytest.mark.parametrize("group", [32, None])
@pytest.mark.parametrize("scale_format", ["fp16", "fp32", "bf16", "fp8"])
def test_fused_scale_search_matches_tensor_reference(
    monkeypatch,
    name: str,
    group: int | None,
    scale_format: str,
) -> None:
    weights = torch.randn(29, 160) * (torch.rand(29, 1) * 3 + 0.1)
    format = FORMATS[name](group=group, scale="mse", scale_fmt=scale_format)
    expected = reference(format, weights, monkeypatch)
    result = format(weights)
    absmax = FORMATS[name](group=group, scale_fmt=scale_format)(weights)
    error = (result.w - weights).square().sum()
    # Summation order differs between the paths, so a near tie may select another candidate.
    agreement = (result.params["scales"] == expected.params["scales"]).float().mean()
    assert agreement > 0.97
    torch.testing.assert_close(error, (expected.w - weights).square().sum(), rtol=1e-3, atol=0)
    assert error <= (absmax.w - weights).square().sum() * (1 + 1e-6)


def test_scale_search_handles_partial_groups_and_offsets(monkeypatch) -> None:
    weights = torch.randn(11, 150) + 2
    format = mk.int(4, group=64, scale="mse", asym=True)
    expected = reference(format, weights, monkeypatch)
    result = format(weights)
    assert result.params["scales"].shape == (11, 3)
    torch.testing.assert_close(
        (result.w - weights).square().sum(),
        (expected.w - weights).square().sum(),
        rtol=1e-3,
        atol=0,
    )


def test_non_finite_weights_are_rejected() -> None:
    weights = torch.randn(8, 64)
    weights[3, 17] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        mk.int(4, group=16)(weights)
    weights[3, 17] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        mk.int(4, group=16, asym=True)(weights)
