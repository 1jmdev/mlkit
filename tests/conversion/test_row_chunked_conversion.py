import pytest
import torch
from torch import nn

import mlkit as mk
from mlkit.conversion import model_conversion

FORMATS = {
    "int4": lambda: mk.int(4, group=16),
    "asymmetric": lambda: mk.int(4, group=16, asym=True),
    "gptq": lambda: mk.gptq(mk.int(4, group=16)),
    "gptq-activation-order": lambda: mk.gptq(mk.nf4(group=16), act_order=True),
    "lattice": lambda: mk.scaled(mk.grid.e8p(), group=None),
    "ldlq-lattice": lambda: mk.ldlq(mk.scaled(mk.grid.e8p(), group=None), step=8),
}


def convert(format, monkeypatch, *, chunked: bool) -> mk.QModel:
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(64, 150, bias=False))
    calibration = [torch.randn(96, 64)]
    with monkeypatch.context() as patch:
        if chunked:
            patch.setattr(model_conversion, "CHUNKED_CONVERSION_ELEMENTS", 1024)
            patch.setattr(model_conversion, "ROW_CHUNK_ELEMENTS", 64 * 40)
        return mk.quantize(model, format(), calib=calibration, cache_dir=None)


@pytest.mark.parametrize("name", FORMATS)
def test_row_chunked_conversion_matches_whole_layer_conversion(name: str, monkeypatch) -> None:
    whole = convert(FORMATS[name], monkeypatch, chunked=False)
    chunked = convert(FORMATS[name], monkeypatch, chunked=True)
    expected, actual = whole.quantized["0"], chunked.quantized["0"]
    assert actual.codec == expected.codec
    assert torch.equal(actual.codes, expected.codes)
    assert torch.equal(actual.params["scales"], expected.params["scales"])
    assert actual.bits == pytest.approx(expected.bits)
    torch.testing.assert_close(actual.w, expected.w, rtol=0, atol=0)
    assert torch.equal(chunked.module[0].weight, whole.module[0].weight)
    assert chunked.layer_reports[0].loss == pytest.approx(whole.layer_reports[0].loss, rel=1e-4)


def test_quantizers_that_are_not_row_separable_convert_whole_layers(monkeypatch) -> None:
    assert not mk.awq(mk.int(4)).row_separable
    assert not mk.incoherent(mk.int(4)).row_separable
    assert mk.gptq(mk.int(4)).row_separable
    assert mk.rtn(mk.nf4()).row_separable
    whole = convert(lambda: mk.awq(mk.int(4, group=16), grid=4), monkeypatch, chunked=False)
    chunked = convert(lambda: mk.awq(mk.int(4, group=16), grid=4), monkeypatch, chunked=True)
    assert torch.equal(chunked.quantized["0"].codes, whole.quantized["0"].codes)


def test_row_separable_quantizer_needs_a_codec_with_row_parameters(monkeypatch) -> None:
    @mk.codec("test_unjoinable_rows")
    def decode(codes, *, scale):
        return codes.float() * scale

    class Unjoinable(mk.Quantizer):
        row_separable = True

        def __call__(self, weight, context=None):
            scale = weight.abs().max() / 7
            codes = (weight / scale).round().to(torch.int8)
            return mk.Q(
                codes=codes,
                params={"scale": scale},
                decode=decode,
                codec="test_unjoinable_rows",
                bits=4 * weight.numel(),
            )

    with pytest.raises(ValueError, match="row_parameters"):
        convert(Unjoinable, monkeypatch, chunked=True)
    assert convert(Unjoinable, monkeypatch, chunked=False).quantized["0"].codes is not None


def test_dense_results_of_row_chunks_are_joined(monkeypatch) -> None:
    class Halves(mk.Quantizer):
        row_separable = True

        def __call__(self, weight, context=None):
            return mk.Q((weight * 2).round() / 2, bits=8 * weight.numel())

    whole = convert(Halves, monkeypatch, chunked=False)
    chunked = convert(Halves, monkeypatch, chunked=True)
    assert torch.equal(chunked.quantized["0"].w, whole.quantized["0"].w)
    assert chunked.bpw == whole.bpw == 8
