# mlkit

An extensible PyTorch quantization library for NVIDIA GPUs. **In development:**
the API and checkpoint format may change before the first stable release.

Quantization formats are ordinary functions and small classes assembled from
building blocks: grids, grouped scaling, codecs and calibration algorithms.
Model conversion, evaluation, checkpoints and packed inference all use the same
format interface, so a new format works with every one of them. mlkit targets
CUDA exclusively; the hot paths are fused Triton kernels.

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
calibration = mk.data("wikitext2", n=16, seq=1024)
evaluation = mk.data("wikitext2", n=40, seq=1024, split="test")

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

codebook = torch.tensor([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
format = mk.scaled(mk.grid.values(codebook, bits=3), group=64, scale="mse")
print(mk.capabilities(format))
quantized = mk.quantize(model, mk.gptq(format), calib=calibration)
```

```text
quantizer           scaled(Grid(values, bits=3, dim=1), group=64, scale='mse')
bits per weight     3.2500
codec               scaled
checkpoint          codes with a registered codec
error feedback      fused scalar kernel at step=1
packed inference    fused kernel
online activations  reference rounding
```

`mk.capabilities` rounds a random matrix and reports which fused kernels,
checkpoint form and inference backend a quantizer qualifies for. A format built
from `mk.scaled` inherits fused encoding, fused GPTQ, portable checkpoints and
packed inference. Use `@mk.quantizer` for a complete custom function, or
subclass `mk.Quantizer` and implement `fit` for formats with learned parameters.
Reading `ctx.H`, `ctx.X` or activation statistics triggers calibration only when
needed. [Building a format](docs/building_formats.md) walks through every block.

## Compare methods, mix precisions, quantize the output head

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

recipe = mk.Recipe(
    weights=mk.gptq(mk.int(4, group=128)),
    head=mk.gptq(mk.int(4, group=128)),
)
quantized = mk.quantize(model, recipe, calib=calibration)
```

Patterns are matched in insertion order. A matching `None` skips the layer.
`weights` covers the linear layers inside the repeated blocks. `head` covers
linear layers outside them, such as the output head, and is off by default. A
tied input embedding takes the rounded values of its head and is stored once.

```python
@mk.preset("w4-head4")
def w4_head4():
    return mk.Recipe(weights=mk.gptq(mk.int(4)), head=mk.gptq(mk.int(4)))


quantized = mk.quantize(model, "w4-head4", calib=calibration)
```

## Inference and checkpoints

```python
quantized = mk.quantize(model, mk.int(4, group=128), calib=None)
inference_model = mk.optimize(quantized, compile=True)
inputs = model.tokenizer("Explain weight quantization.", return_tensors="pt")
tokens = inference_model.generate(**inputs, max_new_tokens=64)
print(model.tokenizer.decode(tokens[0], skip_special_tokens=True))

inference_model.save("checkpoints/int4")
restored = mk.optimize(mk.load("checkpoints/int4"), compile=True)
```

Quantization first produces reconstructed weights for research and evaluation.
`optimize` executes scalar codecs of one to eight bits from their packed codes
with a fused CUDA kernel, including INT2 through INT8, NF4 and scalar codebooks.
A packed output head serves its tied embedding from the same codes. Other
formats retain reconstructed weights. Inputs of more than eight rows, such as a
prompt, reconstruct the weight for an ordinary matrix product.
Compiled Hugging Face generation uses a static cache for decoding. The first
generation includes compilation; subsequent calls reuse the compiled graph.
Checkpoints contain a JSON manifest and safetensors, without executable decoder
code. Custom registered codecs must be available when their checkpoints load.

## Development

```sh
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy
uv run python -m benchmarks.microbenchmarks --filter "algorithms/*"
```

See [the contributor guide](docs/contributing.md) for the package layout,
development conventions and the benchmark suite.

Runnable examples cover [model conversion and generation](examples/quantize_model.py),
[a learned format](examples/learned_codebook.py) and
[lattice and trellis quantization](examples/vector_quantization.py).
Additional examples cover [calibrated recipes](examples/calibrated_recipes.py) and
[layer probes](examples/probe_formats.py). See the [API guide](docs/api.md) for
composition and supported features, and [CUDA measurements](docs/performance.md)
for tested Qwen and Llama workloads.
The optional evaluation extra supports [task evaluation](examples/evaluate_tasks.py)
with lm-evaluation-harness.
