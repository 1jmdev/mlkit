# Building a format

A format is anything that turns an FP32 matrix `[out, in]` into a reconstruction.
mlkit provides building blocks at several levels. Use the highest level that
expresses the format: the more of a format is assembled from blocks, the more
of the fused CUDA paths it inherits. `mk.capabilities(format)` reports exactly
what a format receives.

| Level | You write | You receive |
| --- | --- | --- |
| Grid values | A list of representable values | Fused encoding, fused GPTQ, portable checkpoints, packed inference, fused activation rounding |
| Grid function | A rounding function | Grouped scaling and scale search; reference rounding everywhere |
| Fitted class | `fit` that returns a rounder | Learned parameters with the fused paths of the rounder it returns |
| Complete function | `(weight, context) -> tensor or Q` | Full freedom; reference rounding; dense checkpoints unless it returns codes |
| Codec | A decoder function | Portable checkpoints and trainable codec parameters for custom codes |

## Grids and grouped scaling

A grid describes representable values. `mk.scaled` adds row groups, a scale per
group, scale fitting, scale storage and optional asymmetric offsets.

```python
import torch
import mlkit as mk

values = torch.tensor([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
format = mk.scaled(mk.grid.values(values, bits=3, name="log3"), group=64, scale="mse")
```

A grid with explicit values of at most eight bits is encoded, decoded and
rounded inside GPTQ by fused kernels, and `mk.optimize` executes it from packed
codes. Built-in grids: `mk.grid.int(bits)`, `mk.grid.fp("e2m1")` and the other
named floating formats, `mk.grid.values(values)`, `mk.grid.vector(codebook)`,
`mk.grid.e8p()` and `mk.grid.trellis(L, k)`.

A grid may also be a function. It receives values already divided by their
scale and returns the nearest representable values:

```python
@mk.grid(bits=2)
def ternary(normalized):
    return normalized.round().clamp(-1, 1)


format = mk.scaled(ternary, group=128)
```

A function grid has no code table, so it rounds through tensor operations and
its checkpoint stores the dense reconstruction. Prefer explicit values when the
grid has them.

## A complete function

`@mk.quantizer` wraps a function of the weight and the layer context. Keyword
parameters become configuration: calling the quantizer with keywords only
returns a configured copy.

```python
@mk.quantizer
def sign(weight, context, bits=1):
    scale = weight.abs().mean(1, keepdim=True)
    reconstruction = weight.sign() * scale
    return mk.Q(reconstruction, bits=bits * weight.numel() + 16 * scale.numel())


quantized = mk.quantize(model, sign, calib=None)
```

Returning a bare tensor is allowed; its bit cost is then unknown. `Q.bits` is
the total number of bits of the layer. Side information that is fitted once per
layer, such as a codebook, is declared with `context.add_bits(n)`.

The context supplies calibration statistics on demand: `context.H` is the input
second moment, `context.X` a bounded sample of input rows, and
`context.stat(name, function)` any streaming reduction. Nothing is collected
unless a quantizer reads it.

## A fitted class

Error feedback rounds a layer in column blocks. A format that learns parameters
first must separate fitting from rounding: subclass `mk.Quantizer` and implement
`fit(weight, context)`, returning a function `round(values, columns)`. The
simplest fitted class learns its parameters and delegates to `mk.scaled`:

```python
class LearnedCodebook(mk.Quantizer):
    def __init__(self, bits: int = 3, group: int = 64) -> None:
        self.bits = bits
        self.group = group

    def fit(self, weight, context):
        normalized = weight / weight.abs().amax(1, keepdim=True).clamp_min(1e-12)
        centers = mk.kmeans(normalized.flatten(), 2**self.bits, iters=10)
        context.add_bits(16 * centers.numel())
        grid = mk.grid.values(centers, bits=self.bits)
        rounder = mk.scaled(grid, group=self.group).fit(weight, context)
        return rounder.with_metadata(
            trainable=["scales", "values"], parameter_formats={"values": "fp16"}
        )


quantized = mk.quantize(model, mk.gptq(LearnedCodebook()), calib=calibration)
```

The rounder returned by `mk.scaled(...).fit` carries its fitted state, so GPTQ
still uses the fused kernel. `with_metadata` declares which codec parameters a
fine-tuning pass may train and how they are stored. A class gets a description
from its public attributes, here `LearnedCodebook(bits=3, group=64)`, which is
what reports and checkpoints record.

A rounder may also be written by hand. It receives the values of a contiguous
column slice of the fitted region and returns a tensor or `Q` of the same shape.

## A codec

Codes are portable when a registered decoder can rebuild the weight from them.
A checkpoint stores the codec name, the codes and the tensor parameters; the
decoder code itself is never serialized.

```python
@mk.codec("power_of_two", row_parameters=("row_scales",))
def decode_power_of_two(codes, *, row_scales):
    exponents = (codes & 7).float()
    signs = 1.0 - 2.0 * (codes >> 3).float()
    magnitudes = torch.where(exponents == 0, 0.0, torch.exp2(-exponents))
    return signs * magnitudes * row_scales


@mk.quantizer
def power_of_two(weight, context):
    row_scales = weight.abs().amax(1, keepdim=True).clamp_min(1e-12)
    normalized = (weight / row_scales).abs().clamp_min(2.0**-7)
    exponents = (-normalized.log2()).round().clamp(1, 7)
    codes = (exponents + 8 * (weight < 0)).to(torch.uint8)
    return mk.Q(
        codes=codes,
        params={"row_scales": row_scales},
        decode=decode_power_of_two,
        codec="power_of_two",
        bits=4 * weight.numel() + 16 * row_scales.numel(),
        metadata={"code_bits": 4, "trainable": ["row_scales"]},
    )
```

Codes are unsigned integers below `2**code_bits`; a checkpoint packs them at
that width. `Q.w` decodes on access, so parameters named in `trainable` stay
differentiable for `mk.finetune`. `row_parameters` names the parameters that
hold one entry per weight row. A codec that declares them can be joined from
row chunks.

## Wide layers

An output head with a large vocabulary does not fit in working memory at once.
A quantizer that rounds every row independently declares it:

```python
class LearnedCodebook(mk.Quantizer):
    row_separable = True
```

`mk.quantize` then converts layers above 67 million weights in row chunks and
joins the results. `mk.scaled` formats, `mk.rtn`, `mk.gptq` and `mk.ldlq` around
a row-separable format are row separable. Formats that fit shared state across
rows, such as a codebook learned from the whole matrix, are not.

## Inspect and name it

```python
print(mk.capabilities(LearnedCodebook()))
```

```text
quantizer           LearnedCodebook(bits=3, group=64)
bits per weight     3.2578
codec               scaled
checkpoint          codes with a registered codec
error feedback      fused scalar kernel at step=1
packed inference    fused kernel
online activations  reference rounding; not restorable from a checkpoint
```

A recipe that should be reusable by name is registered as a preset. The factory
may return a quantizer or a complete `mk.Recipe`.

```python
@mk.preset("learned3-gptq")
def learned3_gptq():
    return mk.Recipe(weights=mk.gptq(LearnedCodebook()), head=mk.int(8))


quantized = mk.quantize(model, "learned3-gptq", calib=calibration)
```

## What the fused paths require

| Path | Requirement |
| --- | --- |
| Fused encoding and scale search | `mk.scaled` with a scalar grid of at most 256 values and a group that is a multiple of 8 |
| Fused GPTQ | A rounder from `mk.scaled(...).fit` with a scalar grid of at most 8 bits, and `step=1` |
| Fused LDLQ | A rounder from `mk.scaled(...).fit` with a vector grid of dimension 2, 4, 8 or 16, and `step` equal to that dimension |
| Packed inference | Codec `scaled` or `feedback`, at most 8 bits, original column order, refit regions that hold whole groups |
| Fused packed kernel | Scale groups that are multiples of 16, and rows that hold whole packing words: multiples of 8 codes at 1, 3, 5 and 7 bits, of 4 at 2 and 6 bits, of 2 at 4 bits |
| Fused activation rounding | `mk.scaled` with a scalar grid, `scale="absmax"` and a named scale storage format |

Everything outside these requirements still works through the tensor reference
implementations, which the fused kernels reproduce.
