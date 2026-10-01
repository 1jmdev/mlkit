# Contributing

Use UV for dependencies and Python commands. Write readable, formatted code
directly, with descriptive names and one-line conventional commit messages.
Keep format definitions independent of architecture traversal and inference.

## Package organization

- `src/mlkit/quantization`: representations, protocols, grids, formats, algorithms,
  recipes and numerical operations.
- `src/mlkit/runtime`: architecture adapters, conversion, calibration, block
  passes, inference, transforms, checkpoints and CUDA kernels.
- `src/mlkit/experiments`: dataset preparation, perplexity and experiment tables.

The top-level package exports the public API. Internal imports use the appropriate
subpackage rather than importing from the top-level facade.

## Validation

```sh
uv sync --group dev
uv run ruff check .
uv run mypy --python-version 3.12 src/mlkit
uv run pytest
uv build
```

Model execution tests require CUDA. Tests marked `integration` additionally
download model artifacts. Mathematical reference and bitstream tests can run
without model execution. CI runs those tests, lint and package builds.

Performance results must identify hardware, software versions, tensor shapes,
warmup, measurement protocol and evaluation token budget. Report regressions and
supported execution paths alongside improvements. Never infer inference speed
from checkpoint size or logical bits per weight.
