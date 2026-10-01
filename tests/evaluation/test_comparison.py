import torch
from reference_models import CausalModel

import mlkit as mk


def test_comparison_table_and_sweep(tmp_path) -> None:
    model = CausalModel()
    tokens = torch.randint(19, (2, 12))
    table = mk.compare(model, [mk.int(4, group=4)], data=tokens, calib=None, print_table=False)
    assert len(table) == 2
    assert table[0]["method"] == "baseline"
    assert table[1]["bpw"] == 8
    table.to_csv(tmp_path / "comparison.csv")
    assert "perplexity" in (tmp_path / "comparison.csv").read_text()
    table = mk.sweep(
        model,
        lambda bits: mk.int(bits, group=4),
        bits=[2, 3],
        options={"data": tokens, "calib": None, "print_table": False},
    )
    assert len(table) == 3
