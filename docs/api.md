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

Use `@mk.grid(bits=n, dim=d)` for a rounding grid. Grid bits describe one scalar
or one vector of dimension `d`. `mk.scaled` adds grouping, scale fitting and
optional asymmetric offsets. Built-in grids include integers, named floating
formats, scalar codebooks, vector codebooks and E8P.

```python
codebook = torch.tensor([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
format = mk.scaled(mk.grid.values(codebook, bits=3), group=64, scale="mse")
```

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
Native scalar rounders from `mk.scaled(...).fit(...)` retain CUDA execution
information. Their `with_metadata(...)` method can declare learned parameters
without losing fused GPTQ rounding. Integer grids and scalar codebooks of up to
256 values have this fused path; arbitrary custom rounders keep their own logic.

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

```python
fourth_moment = context.stat("fourth_moment", lambda x: x.pow(4).mean(0), reduce="mean")
```

Statistics are collected on demand. Later blocks collect previously requested
statistics together. Known sibling projections share reductions. With
`sequential=True`, calibration propagates through the quantized prefix.
`sequential=False` enables reusable original-model statistics on disk. Cache keys
include weights, calibration tensors, statistic name and sample-row budget.
Shared block metadata has one host snapshot per calibration batch. Completed
block inputs and targets are released during conversion unless a model pass
requires their history.

Online activation quantizers discover statistic requirements during conversion.
Their collected statistics are frozen before inference. Custom quantizers whose
statistic requirements depend on a later input branch must request those
statistics during calibration.

## Recipes, transforms and passes

A quantizer, a preset string, an ordered pattern map or `Recipe` is accepted by
`quantize`. The first matching pattern wins; `None` skips a layer. A pattern value
may be a `context -> quantizer` selector. A bare quantizer skips output heads;
embeddings are never treated as linear matrices.

```python
recipe = mk.Recipe(
    weights=mk.gptq(mk.int(4, group=128)),
    acts=mk.int(4, group=None),
    kv=mk.int(4, group=64),
    transforms=[mk.rotate("hadamard")],
    passes=[mk.finetune(steps=20, bs=2)],
)
quantized = mk.quantize(model, recipe, calib=calibration)
```

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
| Packed scalar execution | Compatible four-bit INT and scalar-codebook codecs |
| Single-token decoding | Fused packed CUDA matrix-vector kernel |
| Prefill and multiple rows | CUDA reconstruction followed by matrix multiplication |
| Compiled HF generation | Eager prefill and compiled decoding with a static KV cache |
| Other weight formats | Reconstructed dense execution |
| Activation quantization | Online reconstructed activations |
| KV quantization | Post-RoPE per-head reconstruction; eager generation |
| TorchAO export | Optional separate requantization backend |

Activation and KV quantization currently simulate quantization error without
reducing activation or cache tensor storage. Online KV quantization cannot be
combined with compiled generation yet. Packed CUDA kernels are inference-only.
TorchAO INT4 tile-packed exports use BF16 parameters and preserve FP32 buffers.
Output heads remain unquantized. Tile padding can increase storage for very small
layers; `storage_bytes` includes the physical tensors inside quantized subclasses.

Checkpoints contain JSON and safetensors. Built-in scalar, vector, E8P, trellis,
basis and channel-scale codecs store codes and parameters. Custom decoder code
is never serialized: register it with `@mk.codec(name)` in the loading process.
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
