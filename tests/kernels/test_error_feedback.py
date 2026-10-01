import pytest
import torch

import mlkit as mk


def calibrated_layer() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(79)
    weights = torch.randn(71, 96, generator=generator)
    inputs = torch.randn(256, 96, generator=generator)
    return weights, inputs


@pytest.mark.parametrize("refit", [16, 64, None])
@pytest.mark.parametrize("act_order", [False, True])
def test_fused_feedback_matches_torch(refit: int | None, act_order: bool) -> None:
    weights, inputs = calibrated_layer()
    context = mk.Ctx(X=inputs)
    quantization = mk.int(4, group=16)
    expected = mk.gptq(
        quantization, refit=refit, act_order=act_order, backend="torch"
    )(weights, context)
    result = mk.gptq(
        quantization, refit=refit, act_order=act_order, backend="triton"
    )(weights, context)
    torch.testing.assert_close(result.w, expected.w, rtol=1e-5, atol=1e-6)
    assert result.bits == expected.bits
    assert result.codec == "feedback"


@pytest.mark.parametrize(
    "format",
    [mk.nf4(group=16), mk.mxfp4(group=16), mk.int(4, group=16, asym=True)],
)
def test_fused_scalar_codebooks_match_reference_rounding(format) -> None:
    weights, inputs = calibrated_layer()
    context = mk.Ctx(X=inputs)
    expected = mk.gptq(format, backend="torch")(weights, context)
    actual = mk.gptq(format, backend="triton")(weights, context)
    torch.testing.assert_close(actual.w, expected.w, rtol=1e-5, atol=1e-6)
    assert torch.equal(actual.codes, expected.codes)
    assert actual.bits == expected.bits


def test_fitted_class_preserves_fused_rounding_and_metadata() -> None:
    class FittedCodebook(mk.Quantizer):
        def fit(self, weight, context):
            codebook = torch.linspace(-1, 1, 9, device=weight.device)
            context.add_bits(16 * codebook.numel())
            format = mk.scaled(mk.grid.values(codebook, bits=4), group=16)
            return format.fit(weight, context).with_metadata(
                trainable=["scales", "values"], parameter_formats={"values": "fp16"}
            )

    weights, inputs = calibrated_layer()
    reference_context = mk.Ctx(X=inputs)
    fused_context = mk.Ctx(X=inputs)
    expected = mk.gptq(FittedCodebook(), refit=None, backend="torch")(weights, reference_context)
    actual = mk.gptq(FittedCodebook(), refit=None, backend="triton")(weights, fused_context)
    torch.testing.assert_close(actual.w, expected.w, rtol=1e-5, atol=1e-6)
    assert actual.metadata == expected.metadata
    assert actual.metadata["trainable"] == ["scales", "values"]
    assert fused_context.additional_bits == reference_context.additional_bits == 144
