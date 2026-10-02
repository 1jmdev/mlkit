import pytest
import torch
from torch import nn

import mlkit as mk


class Ternary(mk.Quantizer):
    def __init__(self, group: int = 64, damping: float = 0.5) -> None:
        self.group = group
        self.damping = damping
        self.inner = mk.int(2, group=group)
        self._scratch = torch.zeros(1)

    def fit(self, weight, context):
        return self.inner.fit(weight, context)


def test_custom_quantizers_have_a_stable_description() -> None:
    description = repr(Ternary(group=32))
    assert description == (
        "Ternary(group=32, damping=0.5, "
        "inner=scaled(Grid(int2, bits=2, dim=1), group=32, scale='absmax'))"
    )
    assert repr(Ternary(group=32)) == description
    assert "0x" not in description


def test_built_in_descriptions_name_their_configuration() -> None:
    assert repr(mk.nf4()) == "scaled(Grid(nf4, bits=4, dim=1), group=64, scale='absmax')"
    assert repr(mk.mxfp4()) == (
        "scaled(Grid(e2m1, bits=4, dim=1), group=32, scale='absmax', scale_fmt='e8m0')"
    )
    assert repr(mk.int(4, asym=True, scale="mse")).endswith("scale='mse', asym=True)")
    assert repr(mk.scaled(mk.grid.int(4), scale=mk.absmax)).endswith("scale='absmax')")
    assert repr(mk.best_of(mk.int(4), mk.nf4(), by="mse")).startswith("best_of(scaled(")
    assert "0x" not in repr(mk.best_of(mk.int(4), mk.nf4()))


def test_reports_record_the_description_of_a_custom_quantizer() -> None:
    converted = mk.quantize(nn.Sequential(nn.Linear(64, 8)), Ternary(group=32), calib=None)
    assert converted.layer_reports[0].method.startswith("Ternary(group=32")


def test_registered_presets_resolve_wherever_a_recipe_is_expected() -> None:
    @mk.preset("test-ternary-with-head")
    def ternary_with_head() -> mk.Recipe:
        return mk.Recipe(weights=Ternary(group=32), head=mk.int(8, group=32))

    @mk.preset("test-weights-only")
    def weights_only() -> mk.Quantizer:
        return mk.int(3, group=32)

    model = nn.Sequential(nn.Linear(64, 8))
    converted = mk.quantize(model, "test-ternary-with-head", calib=None)
    assert converted.layer_reports[0].method.startswith("Ternary")
    assert mk.quantize(model, "test-weights-only", calib=None).bpw == 3.5
    with pytest.raises(ValueError, match="already defined"):
        mk.preset("test-weights-only")(weights_only)
    with pytest.raises(ValueError, match="already defined"):
        mk.preset("gptq-int4-g128")(weights_only)
    with pytest.raises(ValueError, match="test-weights-only"):
        mk.quantize(model, "undefined-preset", calib=None)


def test_built_in_presets_still_resolve() -> None:
    model = nn.Sequential(nn.Linear(256, 8))
    assert mk.quantize(model, "rtn-int4-g128", calib=None).bpw == 4.125
    assert mk.quantize(model, "nf4-g64", calib=None).bpw == 4.25
