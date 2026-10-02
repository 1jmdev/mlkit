# Contributing

Use UV for dependencies and Python commands. Write readable, formatted code
directly, with descriptive names and one-line conventional commit messages.
Document with function, class and module docstrings; an inline comment is
warranted only where the code would otherwise be misread. Keep format
definitions independent of architecture traversal and inference.

## Package organization

Packages are layered; a package imports only from those listed before it.

| Package | Contents |
| --- | --- |
| `mlkit.kernels` | Triton kernels and their launch wrappers: scalar encoding and decoding, error feedback, Hadamard transforms, lattice and trellis search, packed linear layers |
| `mlkit.quantization` | `Q`, the quantizer protocol, `Ctx` and recipes, with subpackages `operations`, `grids`, `formats`, `codecs` and `algorithms` |
| `mlkit.models` | Model wrappers, architecture adapters, reports and loading |
| `mlkit.calibration` | Token batches, block input capture, streaming statistics and their cache |
| `mlkit.conversion` | Model conversion, block passes, model transforms, activation and KV quantization |
| `mlkit.checkpoints` | The manifest, saving and loading |
| `mlkit.inference` | Packed layers, backend selection, TorchAO export and benchmarking |
| `mlkit.evaluation` | Perplexity, comparisons, layer probes, capability reports and task evaluation |

The top-level package exports the public API. Internal imports use the
defining module rather than the top-level facade. A module must not share its
name with a function its package re-exports. `tests/` mirrors the packages.

Every fused kernel has a tensor reference implementation that it reproduces,
and a test that compares the two. Kernels take matrix dimensions as runtime
arguments, so one compiled kernel serves every layer shape.

## Validation

```sh
uv sync --group dev
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

The suite needs a CUDA device and runs in about twenty seconds. Tests marked
`device_independent` run without one; CI runs those, lint, type checking and
the package build. Tests marked `integration` use downloaded model artifacts.

## Benchmarks

```sh
uv run python -m benchmarks.microbenchmarks --list
uv run python -m benchmarks.microbenchmarks --filter "algorithms/gptq*" "kernels/*"
uv run python -m benchmarks.microbenchmarks --cold-kernel-cache --output cold.json
uv run python -m benchmarks.compare_results before.json after.json
```

The microbenchmarks use synthetic matrices in the layer shapes of the two
development models and cover operations, formats, kernels, algorithms,
calibration, checkpoints and inference. Use them for every optimization: run
the affected group before and after a change and compare the result files.
Results are written to `benchmark_results/`, which is not tracked; reviewed
results are kept in `benchmarks/results/`.

Model-level benchmarks download a model and take minutes. Run them to confirm
a result, not to iterate:

```sh
uv run python -m benchmarks.models.model_evaluation --model Qwen/Qwen2.5-0.5B
uv run python -m benchmarks.models.generation --model Qwen/Qwen2.5-0.5B
uv run python -m benchmarks.models.generation_accuracy
uv run python -m benchmarks.models.feedback_accuracy
```

Performance results must identify hardware, software versions, tensor shapes,
warmup, measurement protocol and evaluation token budget. Report regressions and
supported execution paths alongside improvements. Never infer inference speed
from checkpoint size or logical bits per weight. A laptop GPU changes clock
state between runs, so compare variants within one process, interleaved, as the
generation benchmark does. Judge a quality change by model perplexity on at
least 40,000 evaluation tokens; a per-layer proxy loss does not reveal every
problem.
