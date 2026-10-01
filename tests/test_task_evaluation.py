import sys
from types import SimpleNamespace

import mlkit as mk


def test_task_evaluation_applies_model_batching_options(monkeypatch) -> None:
    captured = {}

    def create_language_model(**options):
        captured["model"] = options
        return "language_model"

    def evaluate(**options):
        captured["evaluation"] = options
        return {"results": {"arc_easy": {"accuracy": 0.5}}}

    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(simple_evaluate=evaluate))
    monkeypatch.setitem(
        sys.modules, "lm_eval.models.huggingface", SimpleNamespace(HFLM=create_language_model)
    )
    model = SimpleNamespace(module=object(), tokenizer=object())
    result = mk.eval(model, ["arc_easy"], batch_size=4, max_length=128, limit=2)
    assert captured["model"]["batch_size"] == 4
    assert captured["model"]["max_length"] == 128
    assert captured["model"]["pretrained"] is model.module
    assert captured["evaluation"] == {"model": "language_model", "tasks": ["arc_easy"], "limit": 2}
    assert result["results"]["arc_easy"]["accuracy"] == 0.5
