# MLKit

MLKit is a PyTorch quantization library under active development. It separates
quantization formats, calibration algorithms, model conversion, and execution
backends so that a new format can be implemented independently of model traversal.

The implementation provides portable packed checkpoints and CUDA
inference, custom format registration, calibration, evaluation, and reproducible
benchmarks. Performance claims will be accompanied by measured results on the
development machine.

## Development

```sh
uv sync
uv run pytest
uv run ruff check .
```

Source code is divided into three packages:

- `src/mlkit/quantization`: quantizer protocols, formats, grids, algorithms and numerical operations.
- `src/mlkit/runtime`: model adapters, conversion, calibration, passes, inference and checkpoints.
- `src/mlkit/experiments`: dataset preparation, evaluation and experiment tables.

CUDA kernels live in `src/mlkit/runtime/kernels`. Tests live in `tests`, and
benchmarks live in `benchmarks`. The library targets Python 3.11 or newer and
PyTorch 2.8 or newer. The public interface remains `import mlkit as mk`.
