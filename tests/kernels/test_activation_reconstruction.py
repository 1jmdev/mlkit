import pytest
import torch

import mlkit as mk

FORMATS = {
    "int3-g32": mk.int(3, group=32),
    "int4-row": mk.int(4, group=None),
    "int4-fp32": mk.int(4, group=32, scale_fmt="fp32"),
    "nf4-g64": mk.nf4(group=64),
    "mxfp4": mk.mxfp4(),
    "int4-asymmetric": mk.int(4, group=32, asym=True),
    "int4-asymmetric-fp32": mk.int(4, group=32, asym=True, scale_fmt="fp32"),
    "int4-bf16": mk.int(4, group=32, scale_fmt="bf16"),
    "int4-fp8": mk.int(4, group=32, scale_fmt="fp8"),
    "e2m1-fp16": mk.scaled(mk.grid.fp("e2m1"), group=32),
}


@pytest.mark.parametrize("name", FORMATS)
def test_fused_activation_reconstruction_matches_fitted_format(name: str) -> None:
    format = FORMATS[name]
    inputs = torch.randn(7, 130, dtype=torch.float16)
    expected = format(inputs.float()).w.half()
    result = format.reconstruct_activations(inputs)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", FORMATS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_activation_reconstruction_is_exact_on_rounding_ties(name: str, dtype) -> None:
    """Half-precision inputs produce exact rounding ties, which approximate division misplaces."""
    format = FORMATS[name]
    generator = torch.Generator(device="cuda").manual_seed(97)
    inputs = torch.randn(512, 2050, dtype=dtype, generator=generator)
    inputs *= torch.rand(512, 1, generator=generator) * 20 + 0.01
    expected = format(inputs.float()).w.to(dtype)
    result = format.reconstruct_activations(inputs)
    mismatched = int((result != expected).sum())
    assert mismatched == 0, f"{mismatched} of {result.numel()} elements differ"
