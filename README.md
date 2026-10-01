# mlkit

An extensible PyTorch quantization library for NVIDIA GPUs. **In development:**
the API and checkpoint format may change before the first stable release.

Quantization formats are ordinary functions. Calibration algorithms, model
conversion, evaluation and packed inference use the same format interface.
CUDA placement is automatic throughout mlkit's model API.

## Installation

Install from this repository into a project with a CUDA-enabled PyTorch installation:

```sh
uv add "mlkit[transformers,datasets] @ git+https://github.com/1jmdev/mlkit.git"
```

Python 3.11 or later and PyTorch 2.8 or later are required. Packed inference and
fused quantization use Triton, which ships with supported Linux CUDA builds of
PyTorch. Hugging Face model loading and datasets are optional dependencies.

## Quantize and evaluate

```python
import mlkit as mk

model = mk.load("Qwen/Qwen2.5-0.5B", dtype="float16")
calibration = mk.data("wikitext2", n=8, seq=256)
evaluation = mk.data("wikitext2", n=16, seq=256, split="test")

quantized = mk.quantize(model, "gptq-int4-g128", calib=calibration)
print(mk.ppl(quantized, data=evaluation, budget="full"))
print(quantized.bpw)
quantized.report()
```

`quantize` returns a separate model and leaves the source weights unchanged.
`bpw` includes codes, scales and declared side information for quantized layers.
`model_bpw` also includes untouched embeddings, output heads, norms and biases.
Formats that return a tensor without declaring bits report an unknown bit cost.

## Define a format

```python
import torch
import mlkit as mk


@mk.grid(bits=4)
def logarithmic_grid(values):
    codebook = torch.tensor([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
    return mk.snap(values, codebook)


format = mk.scaled(logarithmic_grid, group=64, scale="mse")
quantized = mk.quantize(model, mk.gptq(format), calib=calibration)
```

Use `@mk.quantizer` for a complete custom function, or subclass `mk.Quantizer`
and implement `fit` for formats with learned parameters. Reading `ctx.H`,
`ctx.X` or activation statistics triggers calibration only when needed.

## Compare methods and use mixed precision

```python
mk.compare(
    model,
    ["rtn-int4-g128", "gptq-int4-g128", "nf4-g64"],
    calib=calibration,
    data=evaluation,
    budget="full",
)

quantized = mk.quantize(model, {
    "*.self_attn.*": mk.gptq(mk.int(4, group=128)),
    "*.mlp.down_proj": mk.gptq(mk.int(4, group=64)),
    "*.mlp.*": mk.gptq(mk.int(3, group=128)),
}, calib=calibration)
```

Patterns are matched in insertion order. A matching `None` skips the layer.
A single quantizer skips embeddings and the output head by default.

## Inference and checkpoints

```python
quantized = mk.quantize(model, mk.int(4, group=128), calib=None)
inference_model = mk.optimize(quantized)
inputs = model.tokenizer("Explain weight quantization.", return_tensors="pt")
tokens = inference_model.generate(**inputs, max_new_tokens=64)
print(model.tokenizer.decode(tokens[0], skip_special_tokens=True))

inference_model.save("checkpoints/int4")
restored = mk.optimize(mk.load("checkpoints/int4"))
```

Quantization first produces reconstructed weights for research and evaluation.
`optimize` uses packed four-bit scalar codecs when compatible, including INT4
and NF4. Other formats retain reconstructed weights. Packed decoding uses a
fused CUDA kernel; prefill reconstructs weights for matrix multiplication.
Checkpoints contain a JSON manifest and safetensors, without executable decoder
code. Custom registered codecs must be available when their checkpoints load.

## Development

```sh
uv sync --group dev
uv run pytest
uv run ruff check .
```

See [the contributor guide](docs/contributing.md) for development conventions.

Runnable examples cover [model conversion and generation](examples/quantize_model.py),
[a learned format](examples/learned_codebook.py) and
[lattice and trellis quantization](examples/vector_quantization.py).
