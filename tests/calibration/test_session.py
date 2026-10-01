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
