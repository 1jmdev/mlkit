import pytest
import torch
from reference_models import RepeatedModel
from torch import nn

import mlkit as mk
from mlkit.calibration import CalibrationSession


def test_calibration_shares_metadata_and_releases_completed_blocks() -> None:
    class MetadataBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(16, 16)

        def forward(self, hidden_states, position_ids):
            return self.projection(hidden_states) + position_ids.mean()

    class MetadataModel(RepeatedModel):
        def __init__(self):
            super().__init__()
            self.model.layers = nn.ModuleList([MetadataBlock() for _ in range(3)])

        def forward(self, inputs):
            positions = torch.arange(16, dtype=inputs.dtype, device=inputs.device)
            for block in self.model.layers:
                inputs = block(inputs, position_ids=positions)
            return self.lm_head(inputs)

    wrapped = mk.Model(MetadataModel())
    inputs = torch.randn(2, 16)
    session = CalibrationSession(
        wrapped,
        [inputs],
        sequential=True,
        sample_rows=32,
        cache_dir=None,
        need_targets=False,
        retain_history=False,
    )
    session.prepare()
    positions = [session.block_calls(index)[0].kwargs["position_ids"] for index in range(3)]
    assert len({value.untyped_storage().data_ptr() for value in positions}) == 1
    with torch.no_grad():
        expected = wrapped.blocks[0](inputs, position_ids=positions[0].cuda())
    session.propagate(0, wrapped.blocks[0])
    session.release(0)
    assert session.block_calls(0) == []
    torch.testing.assert_close(
        session.block_calls(1)[0].hidden().cuda(), expected, rtol=0, atol=0
    )


def test_selected_calibration_blocks_capture_only_requested_inputs() -> None:
    wrapped = mk.Model(RepeatedModel())
    inputs = torch.randn(2, 16)
    session = CalibrationSession(
        wrapped,
        [inputs],
        sequential=False,
        sample_rows=32,
        cache_dir=None,
        need_targets=False,
        selected_blocks=(1,),
    )
    session.prepare()
    assert session.block_calls(0) == []
    assert session.block_calls(2) == []
    with torch.no_grad():
        expected = wrapped.blocks[0](inputs)
    torch.testing.assert_close(
        session.block_calls(1)[0].hidden().cuda(), expected, rtol=0, atol=0
    )


def count_complete_forwards(model: nn.Module) -> list[bool]:
    completed: list[bool] = []
    model.lm_head.register_forward_hook(lambda *arguments: completed.append(True))
    return completed


def test_sequential_capture_ends_the_forward_after_the_first_block() -> None:
    model = RepeatedModel()
    wrapped = mk.Model(model)
    completed = count_complete_forwards(model)
    batches = [torch.randn(2, 16) for _ in range(4)]
    session = CalibrationSession(
        wrapped,
        batches,
        sequential=True,
        sample_rows=32,
        cache_dir=None,
        need_targets=False,
    )
    session.prepare()
    assert len(completed) == 1
    for index in range(3):
        assert len(session.block_calls(index)) == 4
    for batch, call in zip(batches, session.block_calls(0), strict=True):
        torch.testing.assert_close(call.hidden().cuda(), batch, rtol=0, atol=0)
    assert all(call.hidden().numel() == 0 for call in session.block_calls(1))


def test_early_capture_produces_the_statistics_of_complete_forwards(monkeypatch) -> None:
    calibration = [torch.randn(2, 8, 16) for _ in range(3)]

    def collect(model: nn.Module) -> dict[str, torch.Tensor]:
        observed = {}

        @mk.quantizer
        def record(weight, context):
            observed[context.name] = context.H.clone()
            return mk.Q(weight.round(), bits=8 * weight.numel())

        mk.quantize(model, record, calib=calibration, cache_dir=None)
        return observed

    model = RepeatedModel()
    early = collect(model)
    monkeypatch.setattr(
        CalibrationSession,
        "_later_blocks_repeat_first_block_arguments",
        lambda self: False,
    )
    complete = collect(model)
    assert early.keys() == complete.keys()
    for name, value in early.items():
        torch.testing.assert_close(value, complete[name], rtol=0, atol=0)


def test_blocks_with_distinct_arguments_require_complete_forwards() -> None:
    class IndexedBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(16, 16)

        def forward(self, hidden_states, depth):
            return self.projection(hidden_states) * (1 + depth)

    class IndexedModel(RepeatedModel):
        def __init__(self):
            super().__init__()
            self.model.layers = nn.ModuleList([IndexedBlock() for _ in range(3)])

        def forward(self, inputs):
            for depth, block in enumerate(self.model.layers):
                inputs = block(inputs, depth=depth)
            return self.lm_head(inputs)

    model = IndexedModel()
    completed = count_complete_forwards(model)
    session = CalibrationSession(
        mk.Model(model),
        [torch.randn(2, 16) for _ in range(3)],
        sequential=True,
        sample_rows=32,
        cache_dir=None,
        need_targets=False,
    )
    session.prepare()
    assert len(completed) == 3
    assert [call.kwargs["depth"] for call in session.block_calls(2)] == [2, 2, 2]


@pytest.mark.parametrize(("policy", "on_cuda"), [("cuda", True), ("host", False)])
def test_activation_storage_policy_places_captured_inputs(policy: str, on_cuda: bool) -> None:
    session = CalibrationSession(
        mk.Model(RepeatedModel()),
        [torch.randn(2, 16) for _ in range(2)],
        sequential=True,
        sample_rows=32,
        cache_dir=None,
        need_targets=False,
        storage=policy,
    )
    session.prepare()
    assert all(call.hidden().is_cuda == on_cuda for call in session.block_calls(0))
    session.propagate(0, session.model.blocks[0])
    assert all(call.hidden().is_cuda == on_cuda for call in session.block_calls(1))
