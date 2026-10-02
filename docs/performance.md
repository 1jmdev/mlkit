# CUDA measurements

These development measurements were collected on an NVIDIA RTX 4060 Laptop GPU
with 8 GB VRAM, PyTorch 2.14.1+cu130, CUDA 13.0 and Triton 3.8.0 on October 2,
2026. They describe these workloads and software versions; they are not a
universal throughput claim. A laptop GPU changes clock state between runs:
code that did not change, such as dense FP16 linear layers and trellis decoding,
measured 5 to 20 percent slower in the second microbenchmark run below, so read
differences of that size as noise.

## Microbenchmarks

The [microbenchmark suite](../benchmarks/microbenchmarks) times synthetic
matrices in the layer shapes of Llama 3.2 1B and Qwen2.5 0.5B. Each case reports
the synchronized median after warmup. The baseline is the restructured code
before any optimization; the geometric mean over its 105 cases is 2.23 times
faster now. Selected cases, in milliseconds:

| Case | Before | After |
| --- | ---: | ---: |
| INT4 rounding, group 128, 2048×8192 | 9.74 | 1.52 |
| NF4 rounding, group 64, 2048×8192 | 21.10 | 1.67 |
| INT4 with MSE scale search, 2048×8192 | 97.70 | 5.98 |
| Vector codebook rounding, 256×8, 2048×2048 | 71.43 | 6.90 |
| E8P rounding, 2048×8192 | 51.81 | 37.64 |
| Scalar decode, 2048×8192 | 2.76 | 0.46 |
| Structured orthogonal transform, 2048×8192 | 21.60 | 3.11 |
| Nearest codeword, 1M vectors × 256 codewords | 133.58 | 10.49 |
| Pack 4-bit / 3-bit codes, 16.8M codes | 3.42 / 5.22 | 0.25 / 0.36 |
| Unpack 4-bit / 3-bit codes, 16.8M codes | 16.07 / 16.02 | 0.31 / 0.41 |
| GPTQ INT4, 2048×2048 | 21.00 | 9.61 |
| GPTQ INT4, 8192×2048 | 37.55 | 23.95 |
| GPTQ INT4, 2048×8192 | 385.02 | 103.53 |
| GPTQ INT4, 896×4864 | 100.61 | 32.23 |
| LDLQ E8P, step 8, 2048×2048 | 79.61 | 16.06 |
| Incoherent GPTQ INT4, 2048×2048 | 33.77 | 12.58 |
| AWQ INT4, 20 candidates, 2048×2048 | 94.71 | 63.45 |
| Hessian of 2048 half-precision rows, 8192 wide | 64.39 | 16.17 |
| Sequential GPTQ conversion, 4 synthetic blocks | 689.77 | 280.55 |
| Checkpoint save, 4 layers of 2048×2048 | 32.53 | 7.69 |
| Checkpoint load, 4 layers of 2048×2048 | 273.86 | 68.99 |
| Packed INT4 linear, 2048×8192, 1 row | 0.129 | 0.091 |
| Packed INT4 linear, 2048×8192, 4 rows | 0.265 | 0.122 |
| Packed INT4 stack of 48 Llama-sized layers, 1 row | 3.33 | 2.58 |

The dense FP16 stack takes 4.99 ms for one row. Packed execution of 2-, 3-, 6-
and 8-bit codes runs within 25 percent of the 4-bit kernel. The packed stack
equals the dense stack at eight rows and is slower beyond, where it reconstructs
weights for an ordinary matrix product. Trellis rounding, scalar k-means and the
proxy loss are unchanged.

Kernels take matrix dimensions as runtime arguments, so one compilation serves
every layer shape. With an empty kernel cache, the first GPTQ call on the eight
layer shapes took 26.5 seconds in total before and 0.45 seconds now.

The raw results for the
[baseline](../benchmarks/results/rtx_4060_microbenchmarks_before_optimization.json),
the [current code](../benchmarks/results/rtx_4060_microbenchmarks_after_optimization.json)
and the empty kernel cache
([before](../benchmarks/results/rtx_4060_cold_kernel_cache_before_optimization.json),
[after](../benchmarks/results/rtx_4060_cold_kernel_cache_after_optimization.json))
include every case, first-call times and peak allocations.

```sh
uv run python -m benchmarks.microbenchmarks
uv run python -m benchmarks.microbenchmarks --cold-kernel-cache --filter "algorithms/*"
uv run python -m benchmarks.compare_results before.json after.json
```

## Quantization and perplexity

Calibration uses sixteen WikiText2 windows of 1024 tokens with seed 17.
Evaluation uses forty contiguous WikiText2 test windows of 1024 tokens, which
gives 40,920 scored next-token targets. Conversion time is one synchronized
conversion with compiled kernels, including calibration.

| Method | Qwen2.5 0.5B perplexity | Conversion | Llama 3.2 1B perplexity | Conversion |
| --- | ---: | ---: | ---: | ---: |
| FP16 baseline | 14.45 | — | 11.27 | — |
| RTN INT4, group 128 | 18.55 | 0.32 s | 14.24 | 0.63 s |
| NF4, group 64 | 16.09 | 0.28 s | 12.44 | 0.53 s |
| GPTQ INT4, group 128 | 15.96 | 4.40 s | 12.73 | 10.02 s |
| GPTQ INT4, activation order | 15.70 | 4.38 s | 12.39 | 10.27 s |
| GPTQ NF4, group 64 | 15.33 | 4.42 s | 12.01 | 10.56 s |
| GPTQ INT4 with INT8 head | 15.96 | 4.38 s | 12.73 | 10.35 s |
| GPTQ INT4 with GPTQ INT4 head | 16.26 | 5.20 s | 12.89 | 12.44 s |
| GPTQ INT4 without the token energy limit | 15.97 | 4.09 s | 15.02 | 9.78 s |

The last row shows why calibration limits the energy of each token. In Llama
3.2 1B, one input feature of `model.layers.1.mlp.down_proj` has a second moment
about four million times the median, produced by roughly one token per window.
Unlimited, those tokens dominate the Hessian of that layer, and GPTQ becomes
worse than direct rounding. Qwen has no such tokens and is unaffected.

Quantizing the output head changes the whole-model precision, because the head
and its tied embedding hold a large share of a small model:

| Recipe | Qwen2.5 0.5B bits per weight | Llama 3.2 1B bits per weight |
| --- | ---: | ---: |
| GPTQ INT4 blocks, FP16 head | 7.40 | 6.65 |
| GPTQ INT4 blocks, INT8 head | 5.23 | 4.98 |
| GPTQ INT4 blocks, GPTQ INT4 head | 4.13 | 4.13 |

The raw [Qwen](../benchmarks/results/rtx_4060_qwen_evaluation.json) and
[Llama](../benchmarks/results/rtx_4060_llama_evaluation.json) results include
token budgets and evaluation times. A separate
[layer check](../benchmarks/results/rtx_4060_llama_feedback_accuracy.json)
confirms that fused GPTQ reproduces the tensor reference exactly on the
calibrated layers of the first Llama block.

```sh
uv run python -m benchmarks.models.model_evaluation --model Qwen/Qwen2.5-0.5B
uv run python -m benchmarks.models.model_evaluation --model meta-llama/Llama-3.2-1B
uv run python -m benchmarks.models.feedback_accuracy
```

## Complete generation

Each measurement includes prompt processing, greedy token selection, KV-cache
updates and 64 generated tokens at batch size one. All variants are built and
warmed with three generations, which also compiles them. Seven rounds then run
every variant once, starting from a different variant each round, so a change of
clock state affects all variants alike. The table reports the median, the range
over the rounds, and the median of the per-round speedups over FP16 eager.

INT4 blocks use round-to-nearest with groups of 128 and FP16 scales. Compiled
generation uses eager prefill and a static cache for compiled decoding.

| Model | Execution | Tokens/s | Range | Speedup | Model storage |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5 0.5B | FP16 eager | 75.0 | 73.7–75.3 | 1.00 | 0.92 GiB |
| Qwen2.5 0.5B | FP16 compiled | 158.3 | 148.5–159.1 | 2.12 | 0.92 GiB |
| Qwen2.5 0.5B | Packed INT4 eager | 63.1 | 62.9–63.5 | 0.84 | 0.43 GiB |
| Qwen2.5 0.5B | Packed INT4 compiled | 202.8 | 189.1–204.4 | 2.71 | 0.43 GiB |
| Qwen2.5 0.5B | Packed INT4, INT8 head, compiled | 227.3 | 209.5–230.2 | 3.04 | 0.30 GiB |
| Qwen2.5 0.5B | Packed INT4, INT4 head, compiled | 231.9 | 202.0–237.7 | 3.07 | 0.24 GiB |
| Qwen2.5 0.5B | TorchAO INT4 compiled | 233.4 | 229.8–235.3 | 3.11 | 0.45 GiB |
| Llama 3.2 1B | FP16 eager | 81.7 | 81.2–81.8 | 1.00 | 2.30 GiB |
| Llama 3.2 1B | FP16 compiled | 87.6 | 85.9–87.7 | 1.07 | 2.30 GiB |
| Llama 3.2 1B | Packed INT4 eager | 82.9 | 82.4–83.4 | 1.01 | 0.96 GiB |
| Llama 3.2 1B | Packed INT4 compiled | 137.3 | 133.7–138.5 | 1.69 | 0.96 GiB |
| Llama 3.2 1B | Packed INT4, INT8 head, compiled | 159.9 | 157.9–160.9 | 1.96 | 0.72 GiB |
| Llama 3.2 1B | Packed INT4, INT4 head, compiled | 168.5 | 167.5–170.1 | 2.07 | 0.59 GiB |

The INT4-head rows come from a second run of each model with fewer variants; in
that run TorchAO INT4 measured 227.5 tokens/s on Qwen. TorchAO export performs
its own quantization, so its quality needs an independent evaluation, and it
leaves the output head unquantized. Model storage excludes the KV cache and
temporary workspace.

Packed inference alone is not a guarantee of lower latency. Eager packed Qwen is
slower than eager FP16, because launching a kernel costs more than its small
layers take to execute. Compilation removes that overhead. Larger prefill
batches reconstruct weights before matrix multiplication. Measure your own
prompt lengths and batch sizes.

The raw [Qwen](../benchmarks/results/rtx_4060_qwen_generation.json) and
[Llama](../benchmarks/results/rtx_4060_llama_generation.json) measurements, and
the INT4-head runs for [Qwen](../benchmarks/results/rtx_4060_qwen_generation_head_int4.json)
and [Llama](../benchmarks/results/rtx_4060_llama_generation_head_int4.json),
include every sample and the generated tokens.

```sh
uv run python -m benchmarks.models.generation --model Qwen/Qwen2.5-0.5B
uv run python -m benchmarks.models.generation --model meta-llama/Llama-3.2-1B
uv run python -m benchmarks.models.generation --head-bits 4 \
    --variants dense packed-compiled packed-head-compiled
```

A separate [Llama accuracy check](../benchmarks/results/rtx_4060_llama_generation_accuracy.json)
compares compiled cached logits with eager logits on identical prefixes. Over
32 positions the RMS logit error was 0.0042 for FP16, 0.0046 for packed INT4 and
0.0057 for packed INT4 with a packed INT8 head. FP16 and the packed head each
changed one next-token choice. Small FP16 differences can alter subsequent
greedy sequences; generated text is not expected to be bitwise identical across
execution paths.

```sh
uv run python -m benchmarks.models.generation_accuracy
```

For your own measurements, `mk.benchmark` synchronizes CUDA before and after each
sample. Its peak allocation includes tensors already resident in the process.
Avoid concurrent GPU workloads, include warmup and compilation policy, and report
both logical quantization cost and actual model storage.
