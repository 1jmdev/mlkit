# CUDA measurements

These development measurements were collected on an NVIDIA RTX 4060 Laptop GPU
with 8 GB VRAM, PyTorch 2.14.1+cu130 and CUDA 13.0 on October 1, 2026. They describe
these workloads and software versions; they are not a universal throughput claim.

## Complete generation

Each measurement includes prompt processing, greedy token selection, KV-cache
updates and 64 generated tokens at batch size one. Two complete generations warm
the execution path; five synchronized wall-clock measurements determine the
median. Compilation is excluded. Each backend starts from a fresh model, and
only that model remains resident during measurement.

INT4 uses round-to-nearest with groups of 128 and FP16 scales. Embeddings and the
output head remain FP16. Compiled generation uses eager prefill and a static
cache for compiled decoding.

| Model | Execution | Tokens/s | Registered model storage |
| --- | --- | ---: | ---: |
| Qwen2.5 0.5B | FP16 eager | 79.1 | 0.92 GiB |
| Qwen2.5 0.5B | FP16 compiled | 162.6 | 0.92 GiB |
| Qwen2.5 0.5B | Reconstructed INT4 compiled | 165.0 | 0.92 GiB |
| Qwen2.5 0.5B | Packed INT4 compiled | 198.3 | 0.43 GiB |
| Llama 3.2 1B | FP16 eager | 83.2 | 2.30 GiB |
| Llama 3.2 1B | FP16 compiled | 88.8 | 2.30 GiB |
| Llama 3.2 1B | Reconstructed INT4 compiled | 89.4 | 2.30 GiB |
| Llama 3.2 1B | Packed INT4 compiled | 121.8 | 0.96 GiB |

Packed Llama generation peaked at approximately 1.00 GiB of allocated model,
cache and workspace tensors. Model storage excludes the KV cache and temporary
workspace. Conversion can use more memory because it preserves the source model.

Packed inference alone is not a guarantee of lower latency. Earlier eager Qwen
runs showed overhead exceeding the weight-bandwidth savings. Compilation matters
for this workload, and larger prefill batches reconstruct weights before matrix
multiplication. Measure your own prompt lengths and batch sizes.

The raw [Qwen measurements](../benchmarks/results/rtx_4060_qwen_generation.json)
and [Llama measurements](../benchmarks/results/rtx_4060_llama_generation.json)
include prompt lengths, latency distributions, allocation peaks and software
versions. Reproduce them with:

```sh
uv run python benchmarks/generation.py --compile
uv run python benchmarks/generation.py --model meta-llama/Llama-3.2-1B --compile
```

A separate [Llama accuracy check](../benchmarks/results/rtx_4060_llama_generation_accuracy.json)
compares compiled cached logits with eager logits on identical prefixes. Over
32 positions, packed INT4 had an RMS logit error of 0.0047 and identical next-token
choices. Compiled FP16 had an RMS error of 0.0042 and one changed next-token choice.
Small FP16 differences can alter subsequent greedy sequences; generated text is
not expected to be bitwise identical across execution paths.

```sh
uv run python benchmarks/generation_accuracy.py
```

An optional [TorchAO INT4 export](../benchmarks/results/rtx_4060_qwen_torchao_generation.json)
measured 222.3 tokens/s on Qwen with BF16 parameters and 0.45 GiB of registered
model storage. Export performs separate quantization, so its quality needs an
independent evaluation. It preserves FP32 model buffers and leaves output heads
unquantized.

```sh
uv run python benchmarks/generation.py --compile --backends torchao-int4
```

## Quantization and perplexity checks

This short Qwen development check uses eight calibration windows of 256 tokens
and sixteen contiguous WikiText2 test windows, with seed 17. There are 4,080
scored next-token targets. WikiText is tokenized as a single joined corpus.
These are short-context development results, not full benchmark-protocol scores.

| Method | Quantized-layer bpw | Perplexity | Conversion time |
| --- | ---: | ---: | ---: |
| FP16 baseline | 16.000 | 22.26 | — |
| RTN INT4, group 128 | 4.125 | 28.35 | 0.50 s |
| NF4, group 64 | 4.250 | 25.83 | 0.62 s |
| GPTQ INT4, group 128 | 4.125 | 25.43 | 5.57 s |

The [raw evaluation results](../benchmarks/results/rtx_4060_qwen_evaluation.json)
include token budgets and evaluation times. Conversion timings depend on Triton
compilation-cache state. Use a full evaluation corpus and a representative
calibration budget before reporting a model's final quality.

The same short [Llama evaluation](../benchmarks/results/rtx_4060_llama_evaluation.json)
measured perplexities of 18.35 for FP16, 23.60 for RTN INT4, 20.49 for NF4 and
23.06 for GPTQ INT4. GPTQ conversion took 13.90 seconds with cached kernels.
Lower bit cost does not imply a better quality result; compare methods on the
same tokens and calibration budget.

```sh
uv run python benchmarks/model_evaluation.py
```

The one-file learned-codebook example was also run on Qwen. Native fitted scalar
rounders reduced GPTQ conversion from 134.7 to 60.4 seconds and its incoherent
variant from 136.9 to 49.9 seconds, with unchanged reported perplexities of 28.98
and 27.62 on the 508-target short check. A 256×1024 matrix check matched all
reference codes exactly after using round-to-nearest FP32 division in the kernel.

Weighted scalar fitting now uses binary search and fixed-order segmented
reductions. The [fitting measurement](../benchmarks/results/rtx_4060_scalar_codebook_fitting.json)
took 57.8 ms for one million samples, sixteen centers and ten Lloyd iterations.

```sh
uv run python examples/learned_codebook.py --model Qwen/Qwen2.5-0.5B
uv run python benchmarks/codebook_fitting.py
```

For your own measurements, `mk.benchmark` synchronizes CUDA before and after each
sample. Its peak allocation includes tensors already resident in the process.
Avoid concurrent GPU workloads, include warmup and compilation policy, and report
both logical quantization cost and actual model storage.
