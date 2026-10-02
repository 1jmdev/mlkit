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


def vector_formats() -> dict[str, tuple[mk.Quantizer, int]]:
    generator = torch.Generator(device="cuda").manual_seed(83)
    octets = torch.randn(256, 8, generator=generator)
    quadruples = torch.randn(600, 4, generator=generator)
    return {
        "lattice": (mk.scaled(mk.grid.e8p(), group=None), 8),
        "grouped-lattice": (mk.scaled(mk.grid.e8p(), group=32), 8),
        "octets": (mk.scaled(mk.grid.vector(octets), group=None), 8),
        "quadruples": (mk.scaled(mk.grid.vector(quadruples), group=16), 4),
    }


@pytest.mark.parametrize("name", ["lattice", "grouped-lattice", "octets", "quadruples"])
@pytest.mark.parametrize("refit", [32, None])
def test_fused_vector_feedback_matches_reference_rounding(name: str, refit: int | None) -> None:
    format, dimension = vector_formats()[name]
    weights, inputs = calibrated_layer()
    context = mk.Ctx(X=inputs)
    expected = mk.ldlq(format, step=dimension, refit=refit, backend="torch")(weights, context)
    actual = mk.ldlq(format, step=dimension, refit=refit, backend="triton")(weights, context)
    assert actual.codec == expected.codec == "vector_feedback"
    assert actual.codes.dtype == expected.codes.dtype
    assert actual.bits == pytest.approx(expected.bits)
    assert actual.metadata == expected.metadata
    # Feedback sums are ordered differently, so a rare near tie may select another codeword.
    assert (actual.codes == expected.codes).float().mean() > 0.99
    hessian = context.H
    torch.testing.assert_close(
        mk.proxy_loss(weights, actual.w, hessian),
        mk.proxy_loss(weights, expected.w, hessian),
        rtol=1e-2,
        atol=0,
    )


def test_fused_vector_feedback_requires_a_step_of_the_grid_dimension() -> None:
    format, _ = vector_formats()["lattice"]
    weights, inputs = calibrated_layer()
    context = mk.Ctx(X=inputs)
    with pytest.raises(ValueError, match="step equal to its dimension"):
        mk.ldlq(format, step=16, refit=None, backend="triton")(weights, context)
    result = mk.ldlq(format, step=16, refit=None)(weights, context)
    assert result.codec == "vector_feedback"


def test_fused_vector_feedback_reduces_the_loss_of_direct_rounding() -> None:
    format, dimension = vector_formats()["lattice"]
    weights, inputs = calibrated_layer()
    context = mk.Ctx(X=inputs)
    direct = format(weights, context)
    result = mk.ldlq(format, step=dimension, refit=None)(weights, context)
    hessian = context.H
    assert mk.proxy_loss(weights, result.w, hessian) < mk.proxy_loss(weights, direct.w, hessian)
