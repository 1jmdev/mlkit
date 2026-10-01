import pytest
import torch

import mlkit as mk

FORMATS = {
    "int3-g32": mk.int(3, group=32),
    "int4-row": mk.int(4, group=None),
    "nf4-g64": mk.nf4(group=64),
    "mxfp4": mk.mxfp4(),
    "int4-asymmetric": mk.int(4, group=32, asym=True),
    "int4-bf16": mk.int(4, group=32, scale_fmt="bf16"),
    "int4-fp8": mk.int(4, group=32, scale_fmt="fp8"),
}


@pytest.mark.parametrize("name", FORMATS)
def test_fused_activation_reconstruction_matches_fitted_format(name: str) -> None:
    format = FORMATS[name]
    inputs = torch.randn(7, 130, dtype=torch.float16)
    expected = format(inputs.float()).w.half()
    result = format.reconstruct_activations(inputs)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
