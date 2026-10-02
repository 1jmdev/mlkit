"""Loading of Hugging Face causal models and mlkit checkpoints onto CUDA."""

from pathlib import Path
from typing import Any

import torch
from torch import nn

from mlkit.models.model import Model


def load(
    name: str | Path,
    dtype: str | torch.dtype = "auto",
    **options: Any,
) -> Model:
    from mlkit.checkpoints import MANIFEST_FILE, load_checkpoint

    path = Path(name)
    if (path / MANIFEST_FILE).is_file():
        return load_checkpoint(path, **options)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("Hugging Face loading requires uv add 'mlkit[transformers]'") from error
    selected_dtype = getattr(torch, dtype) if isinstance(dtype, str) and dtype != "auto" else dtype
    module: nn.Module = AutoModelForCausalLM.from_pretrained(
        str(name), dtype=selected_dtype, **options
    )
    module.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(str(name), trust_remote_code=False)
    return Model(module, tokenizer, name=str(name))
