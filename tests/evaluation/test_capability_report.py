import pytest
import torch

import mlkit as mk


@mk.quantizer
def sign(weight, context):
    return weight.sign() * weight.abs().mean()


class Ternary(mk.Quantizer):
    def fit(self, weight, context):
        grid = mk.grid.values(torch.tensor([-1.0, 0.0, 1.0]), bits=2)
        return mk.scaled(grid, group=64).fit(weight, context)


def test_scalar_format_qualifies_for_every_fused_path() -> None:
    report = mk.capabilities(mk.int(4, group=128))
    assert report.bits_per_weight == 4.125
    assert report.codec == "scaled"
    assert report.checkpoint == "codes with a registered codec"
    assert report.error_feedback == "fused scalar kernel at step=1"
    assert report.packed_inference == "fused kernel"
    assert report.online_activations == "fused kernel"
    assert "bits per weight     4.1250" in str(report)


def test_custom_class_built_from_blocks_inherits_fused_paths() -> None:
    report = mk.capabilities(Ternary())
    assert report.quantizer == "Ternary()"
    assert report.bits_per_weight == 2.25
    assert report.error_feedback == "fused scalar kernel at step=1"
    assert report.packed_inference == "fused kernel"
    assert report.online_activations.startswith("reference rounding")


def test_function_quantizer_reports_reference_paths() -> None:
    report = mk.capabilities(sign)
    assert report.bits_per_weight is None
    assert report.codec is None
    assert report.checkpoint == "dense reconstruction"
    assert report.error_feedback.startswith("reference rounding")
    assert report.packed_inference == "no; dense weights"
    assert "unknown" in str(report)


@pytest.mark.parametrize(
    ("quantization", "feedback", "packed"),
    [
        (
            mk.scaled(mk.grid.e8p(), group=None),
            "fused vector kernel at step=8",
            "no; dense weights",
        ),
        (mk.int(12), "reference rounding with the fitted rounder", "no; dense weights"),
        (
            mk.gptq(mk.int(4)),
            "not an inner format; it does not implement Quantizer.fit",
            "fused kernel",
        ),
        (
            mk.awq(mk.int(4)),
            "not an inner format; it does not implement Quantizer.fit",
            "no; dense weights",
        ),
    ],
)
def test_capabilities_of_built_in_quantizers(quantization, feedback: str, packed: str) -> None:
    report = mk.capabilities(quantization)
    assert report.error_feedback == feedback
    assert report.packed_inference == packed


def test_layout_that_splits_a_packing_word_is_reported() -> None:
    report = mk.capabilities(mk.int(3, group=50), shape=(16, 250))
    assert report.packed_inference.startswith("packed storage with dense execution")


def test_invalid_reconstruction_is_rejected() -> None:
    @mk.quantizer
    def truncated(weight, context):
        return weight[:, :-1]

    with pytest.raises(ValueError, match="invalid reconstruction"):
        mk.capabilities(truncated)
