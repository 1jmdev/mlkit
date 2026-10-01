"""Optional language-model task evaluation through lm-evaluation-harness."""

from collections.abc import Sequence
from typing import Any

from mlkit.models.model import Model


def eval(
    model: Model,
    tasks: Sequence[str],
    *,
    batch_size: int | str = 1,
    max_length: int | None = None,
    **options: Any,
) -> dict[str, Any]:
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as error:
        raise ImportError("task evaluation requires uv add 'mlkit[evaluation]'") from error
    language_model = HFLM(
        pretrained=model.module, tokenizer=model.tokenizer,
        batch_size=batch_size, max_length=max_length,
    )
    result = simple_evaluate(model=language_model, tasks=list(tasks), **options)
    if result is None:
        raise RuntimeError("task evaluation returned no results")
    return result
