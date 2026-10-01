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

Source code lives in `src/mlkit`, tests in `tests`, and runnable examples in
`examples`. The library targets Python 3.11 or newer and PyTorch 2.8 or newer.
