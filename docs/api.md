# API guide

mlkit places model weights and input tensors on CUDA automatically. Raw PyTorch
operations in custom formats should allocate tensors on the input tensor's device.
Model conversion preserves the source weights; `optimize(inplace=True)` explicitly
permits replacing layers in the supplied model.

## Quantizer protocol

A quantizer receives an FP32 matrix `[out, in]` and a `Ctx`. It returns a
reconstruction tensor or `Q`. The same protocol accepts activation matrices
`[tokens, in]` and flattened per-head key/value matrices.

```python
import torch
import mlkit as mk


@mk.quantizer
def uniform_quantizer(weight, context, bits=4):
    maximum = 2 ** (bits - 1) - 1
    scales = mk.absmax(weight, qmax=maximum).half().float()
    codes = (weight / scales).round().clamp(-maximum - 1, maximum)
    return mk.Q(codes * scales, bits=bits * weight.numel() + 16 * scales.numel())


int3 = uniform_quantizer(bits=3)
```

A bare tensor has unknown bit cost. `Q.bits` is the total logical number of bits,
including declared side information. `context.add_bits(n)` adds information once
per layer call, including parameters fitted before column rounding.

`mk.capabilities(quantizer)` rounds a random matrix and reports the bit cost,
codec, checkpoint form, error-feedback path, packed inference path and online
activation path of a quantizer. [Building a format](building_formats.md)
describes each building block and what the fused paths require.

Use `@mk.grid(bits=n, dim=d)` for a rounding grid. Grid bits describe one scalar
or one vector of dimension `d`. `mk.scaled` adds grouping, scale fitting and
optional asymmetric offsets. Built-in grids include integers, named floating
formats, scalar codebooks, vector codebooks and E8P.

```python
codebook = torch.tensor([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
format = mk.scaled(mk.grid.values(codebook, bits=3), group=64, scale="mse")
```

Scalar grids with explicit values of at most 256 entries are fitted, encoded and
decoded by fused CUDA kernels that reproduce the tensor reference bit for bit.
`mk.grid.values(values, name=...)` names a grid in reports.

Groups are row-aligned. Partial final groups have their own scale. `group=None`
means one scale per row. Scale fitting accepts `"absmax"`, `"mse"` or a callable;
scale storage accepts `"fp16"`, `"bf16"`, `"fp8"`, `"e8m0"` or another grid.
Declared scale storage is rounded before reconstruction and counted in `bpw`.
Built-in vector grids keep their shared codebook fixed during fine-tuning.
Learned formats can explicitly declare per-layer codebooks as trainable and must
count their storage in `Q.bits` or `context.add_bits`.

## Fitting and algorithms

Subclass `mk.Quantizer` when parameters must be fitted before error feedback.
Implement `fit(weight, context)` and return `round_columns(values, columns)`.
`columns` is a slice relative to the fitted region. The rounder may receive a
single column or a vector-width block. The default `__call__` fits and rounds
the whole matrix.
Native rounders from `mk.scaled(...).fit(...)` retain CUDA execution
information. Their `with_metadata(...)` method can declare learned parameters
without losing fused rounding. Integer grids and scalar codebooks of up to 256
values are rounded inside GPTQ by a fused kernel. E8P and vector codebooks of
dimension 2, 4, 8 or 16 are rounded inside LDLQ by a fused kernel when `step`
equals the grid dimension. Arbitrary custom rounders keep their own logic.

A `mk.Quantizer` subclass is described in reports by its class name and public
attributes. Setting `row_separable = True` declares that rows are rounded
independently, which lets layers above 67 million weights be converted in row
chunks. `mk.scaled`, and `rtn`, `gptq` and `ldlq` around it, are row separable.

| Wrapper | Purpose |
| --- | --- |
| `rtn(format)` | Direct rounding; a bare format already follows this path |
| `gptq(format, refit=128, act_order=False)` | Single-column Hessian error feedback |
| `ldlq(format, step=8, refit=None)` | Block Hessian error feedback |
| `awq(format, grid=20, shared=False)` | Activation-aware channel scale search |
| `incoherent(format, train_signs=False)` | Orthogonal basis changes around quantization |
| `best_of(*formats)` | Select the lowest Hessian proxy loss per layer |

```python
format = mk.scaled(mk.grid.e8p(), group=None)
quantizer = mk.incoherent(mk.ldlq(format, step=8, refit=None))
```

Place changes of basis outside error feedback and rounding. Shared AWQ jointly
searches q/k/v and gate/up siblings using the same configured quantizer. A custom
format under GPTQ must accept the column slices it receives or implement `fit`.

Incoherence uses deterministic structured orthogonal transforms. Available
Hadamard factors are combined with an orthogonal cosine transform when a width
cannot be factored into supported Hadamard sizes. Stored trainable sign vectors
are counted; deterministic signs reconstructed from seeds have zero logical cost.

## Calibration context

| Context field | Value |
| --- | --- |
| `name`, `module`, `block`, `block_idx` | Layer identity and architecture location |
| `H` | FP32 input second moment, `E[x xᵀ]` |
| `X` | Bounded sample of input rows |
| `act_absmean`, `act_absmax` | Per-channel activation magnitudes |
| `siblings` | Names of projections with a recognized shared input |
| `rng` | Layer-name-derived generator |
| `cache` | Shared per-run codebooks and bookkeeping |
| `replace(H=...)` | A context with overridden statistics |
| `derive(provider)` | A context for transformed weights with lazily computed statistics |

```python
fourth_moment = context.stat("fourth_moment", lambda x: x.pow(4).mean(0), reduce="mean")
```

An algorithm that permutes, rotates or rescales the weight columns gives its
inner quantizer `context.derive(provider)`. The provider receives
`(name, function, reduction)` and returns the statistic in the transformed
basis; it is called only for statistics the inner quantizer reads.

Statistics are collected on demand. Later blocks collect previously requested
statistics together. Known sibling projections share reductions. With
`sequential=True`, calibration propagates through the quantized prefix.
`sequential=False` enables reusable original-model statistics on disk. Cache keys
include weights, calibration tensors, statistic name and sample-row budget.
Shared block metadata has one host snapshot per calibration batch. Completed
block inputs and targets are released during conversion unless a model pass
requires their history.

A few tokens of a transformer carry activations orders of magnitude larger than
all others. Left alone they dominate every second moment, and error feedback
then trades the accuracy of ordinary tokens for theirs. `quantize` and `probe`
therefore scale each calibration token down to at most `token_energy_limit`
times the median token energy of its batch; the default is 100 and `None`
disables the limit. On Llama 3.2 1B, GPTQ INT4 reaches a perplexity of 12.73
with the limit and 15.02 without it, where direct rounding reaches 14.24.

Captured block inputs stay on CUDA while a quarter of the device memory, and at
least one gibibyte, remains free; otherwise they move to host memory. `calibration_storage="cuda"` or `"host"`
forces either placement. Second moments of half-precision activations are
multiplied on tensor cores in TF32, which changes a Hessian by a few parts in
100,000; `mlkit.calibration.statistics.TENSOR_FLOAT_PRODUCTS = False` selects
FP32 products.

Online activation quantizers discover statistic requirements during conversion.
Their collected statistics are frozen before inference. Custom quantizers whose
statistic requirements depend on a later input branch must request those
statistics during calibration.

## Recipes, transforms and passes

A quantizer, a preset string, an ordered pattern map or `Recipe` is accepted by
`quantize`. The first matching pattern wins; `None` skips a layer. A pattern value
may be a `context -> quantizer` selector. `weights` applies to the linear layers
inside the repeated blocks.

```python
recipe = mk.Recipe(
    weights=mk.gptq(mk.int(4, group=128)),
    head=mk.gptq(mk.int(4, group=128)),
    acts=mk.int(4, group=None),
    kv=mk.int(4, group=64),
    transforms=[mk.rotate("hadamard")],
    passes=[mk.finetune(steps=20, bs=2)],
)
quantized = mk.quantize(model, recipe, calib=calibration)
```

`head` applies to the linear layers outside the repeated blocks, such as the
output head of a language model, and accepts a quantizer, a selector or a
pattern map. It is `None` by default, which leaves those layers dense. Head
layers are rounded after every block, with statistics from complete forward
passes of the calibration model. A parameter that shares the storage of a
rounded head, such as a tied input embedding, takes the rounded values;
`QModel.tied_weights` records it and checkpoints store the matrix once, as
codes. On the two measured models an INT8 head leaves perplexity unchanged and
a GPTQ INT4 head costs 0.16 to 0.30; see [CUDA measurements](performance.md).

Built-in preset names follow `rtn-int4-g128`, `gptq-int4-g128`, `awq-int4-g128`,
`nf4-g64`, `mxfp4-g32` and `rtn-w4a4`. `@mk.preset(name)` registers a factory
that returns a quantizer or a `Recipe`; the name is then accepted wherever a
recipe is.

`fuse_norms` folds recognized normalization gains and biases into their readers.
`rotate` supports recognized RMS-normalized residual architectures and installs
online transforms on attention and feed-forward outputs. `smooth` balances
individual projections with an online inverse scale; it does not implement a
shared inverse scale folded into the preceding normalization.

A block pass receives `(block, context)`. `context.inputs`, `targets`, `qparams`,
`fp_block`, `rng` and `forward(block, inputs)` support reconstruction training.
Built-in `finetune` trains codec parameters and normalization parameters, then
rounds declared parameter storage and restores cached linear weights. Codes remain
fixed. `@mk.block_pass` makes keyword configuration reusable. `@mk.model_pass`
exposes a lower-level `(model, calibration_session)` callback after conversion.
`QModel.pass_reports` records each block pass's timing, trainable element count
and logged initial/final losses. Layer reports are refreshed after the passes;
their proxy losses use the fixed calibration statistics. Pass reports survive
checkpoint save and load.

Built-in block discovery recognizes Llama, Qwen, Mistral, Gemma, GPT-2, GPT-NeoX
and OPT layouts. Transform and KV support are narrower than block discovery;
unsupported transform sites raise explicit errors. `@mk.adapter(model_type)`
registers a custom architecture adapter.

## Evaluation

`data` accepts WikiText2, C4, RedPajama or text strings. Without a tokenizer it
returns a deferred data source that binds to the model during conversion or
evaluation. Calibration windows are seeded; evaluation windows are contiguous.
WikiText is joined and tokenized once as a single corpus.
The `wikitext2` alias loads [Salesforce's maintained dataset](https://huggingface.co/datasets/Salesforce/wikitext).

`ppl(budget="fast")` measures up to 20,000 next-token targets. Full WikiText
uses all complete test windows, with length 2048 by default. Full C4 uses 256
contiguous validation windows. This bounded C4 protocol is not the entire C4
validation set or a randomly sampled document protocol. Supply prepared token
batches to reproduce a different evaluation protocol exactly.

The first token in each window, padding, and labels of `-100` are excluded.
`return_details=True` reports the actual token count, dataset, maximum evaluated
window length, negative log-likelihood and elapsed time. Custom data determines
the available budget; a short custom batch is not expanded to the full dataset.

`compare` evaluates a baseline and every recipe. `probe` measures Hessian proxy
loss on selected layers without model conversion. `sweep` evaluates a Cartesian
parameter grid. Returned `Table` objects support indexing and CSV export.
`eval(tasks=[...])` integrates the optional lm-evaluation-harness dependency.
Its `batch_size` and `max_length` options configure the model adapter; options
such as `limit` configure the evaluation run.

## Execution and storage

```python
inference = mk.optimize(quantized, compile=True)
result = inference.generate(**tokens, max_new_tokens=64)
inference.save("checkpoints/experiment")
restored = mk.optimize(mk.load("checkpoints/experiment"), compile=True)
```

| Feature | Current behavior |
| --- | --- |
| Packed scalar execution | Scalar-grid codecs of one to eight bits, integer or codebook |
| Decoding of up to eight rows | Fused packed CUDA matrix-vector kernel |
| Prefill and more rows | CUDA reconstruction followed by matrix multiplication |
| Packed output head | Fused kernel; a tied embedding reads its rows from the same codes |
| Compiled HF generation | Eager prefill and compiled decoding with a static KV cache |
| Other weight formats | Reconstructed dense execution |
| Activation quantization | Online reconstructed activations |
| KV quantization | Post-RoPE per-head reconstruction; eager generation |
| TorchAO export | Optional separate requantization backend |

Activation and KV quantization currently simulate quantization error without
reducing activation or cache tensor storage. Online KV quantization cannot be
combined with compiled generation yet. Packed CUDA kernels are inference-only.
A packed layer whose rows or scale groups split a packing word keeps its packed
storage and reconstructs the dense weight on every call; `mk.capabilities`
reports this. `optimize(cache_dense=True)` keeps reconstructed weights resident
for workloads dominated by prefill.
TorchAO INT4 tile-packed exports use BF16 parameters, preserve FP32 buffers and
leave output heads unquantized. Tile padding can increase storage for very small
layers; `storage_bytes` includes the physical tensors inside quantized subclasses.

`mk.benchmark(operation, warmup=10, repetitions=50)` synchronizes CUDA around
every sample and reports median, minimum, 95th percentile and peak allocation.
It runs under `torch.inference_mode`; pass `inference_mode=False` to time an
operation that trains parameters.

Checkpoints contain JSON and safetensors. Built-in scalar, vector, E8P, trellis,
basis and channel-scale codecs store codes and parameters. Custom decoder code
is never serialized: register it with `@mk.codec(name)` in the loading process.
`@mk.codec(name, row_parameters=(...))` additionally names the parameters that
hold one entry per weight row, which lets a row-separable quantizer use the
codec for layers converted in row chunks. Loading reads tensors straight onto
the GPU and builds a Hugging Face architecture without initializing weights.
Unregistered formats save dense reconstructions. Built-in scalar activation and
KV recipes round-trip; custom online callables need an explicit deployment recipe.
Tied dense weights and identical codec parameters are stored once.

`bpw` covers quantized layers and declared side information; `model_bpw` includes
untouched parameters. `storage_bytes` measures distinct registered model tensor
storages, excluding temporary workspace, Python codec state and KV caches.
Checkpoint `tensor_bytes` measures actual stored tensors, excluding JSON headers
and tokenizer files. These values answer different questions.

Trellis quantization implements exact free-start Viterbi search with computed
1MAD Gaussian codes, bounded traceback memory and fused CUDA search for `L <= 12`.
Initial states add `(L - k) / tile²` bits per weight, plus scales. Larger state
spaces use the PyTorch reference search. Tail-biting, HYB/3INST codes, fused
E8P/trellis inference, 70B model offloading and MoE statistics remain future work.
