import argparse
import copy
import itertools
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as functional
from datasets import load_dataset
from torch import nn
from torch.utils.checkpoint import checkpoint

import mlkit as mk
from mlkit.calibration.session import CalibrationSession
from mlkit.calibration.statistics import BlockStatistics
from mlkit.conversion.block_passes import CodecLinear
from mlkit.conversion.model_conversion import convert_layer
from mlkit.evaluation.decision_stability import conversion_monitor
from mlkit.models.module_utilities import extract_hidden


class PrefixCalls:
    def __init__(self, model, templates, index, batches, cache_mib):
        self.model = model
        self.blocks = model.blocks[:index]
        self.templates = templates
        self.index = index
        self.batches = batches
        self.cache = {}
        self.capacity = (cache_mib * 2**20) // (
            batches[0]["input_ids"].numel() * model.config.hidden_size * 2
        )

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return (self.batch([position]) for position in range(*index.indices(len(self))))
        return self.batch([index])

    @torch.no_grad()
    def batch(self, indices):
        if all(index in self.cache for index in indices):
            hidden = torch.cat([self.cache[index] for index in indices])
        else:
            tokens = torch.cat([self.batches[index]["input_ids"] for index in indices]).cuda()
            hidden = self.model.module.model.embed_tokens(tokens)
            for index, block in enumerate(self.blocks):
                hidden = extract_hidden(
                    self.templates[index].with_hidden(hidden).run(block)
                )
            for position, index in enumerate(indices):
                if index not in self.cache and len(self.cache) < self.capacity:
                    self.cache[index] = hidden[position : position + 1].clone()
        return self.templates[self.index].with_hidden(hidden)


class GSQLinear(nn.Module):
    def __init__(self, original, quantized, bits):
        super().__init__()
        self.bias = original.bias
        self.dtype = original.weight.dtype
        self.bits = bits
        self.quantized = quantized
        self.register_buffer("initial_codes", quantized.codes)
        self.register_buffer("grid", quantized.params["values"].to(self.dtype))
        self.scales = nn.Parameter(quantized.params["scales"].float().clone())
        if bits == 2:
            self.register_buffer("candidates", self.grid[:, None, None])
            prior = -0.5 * (self.candidates - self.grid[self.initial_codes.long()]).square()
        else:
            shifts = torch.arange(-2, 3, device="cuda")[:, None, None]
            self.register_buffer(
                "candidates",
                self.grid[(self.initial_codes.long()[None] + shifts).clamp(0, len(self.grid) - 1)],
            )
            prior = -0.5 * torch.arange(-2, 3, device="cuda", dtype=self.dtype).square()
            prior = prior[:, None, None].expand(5, *self.initial_codes.shape)
        self.logits = nn.Parameter(
            0.01 * (torch.randn_like(prior) + 6 * (prior - prior.mean(0)))
        )
        self.temperature = 2.0
        self.sharpness = 100.0

    def soft_weight(self, logits, scales):
        uniform = torch.rand_like(logits)
        noise = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
        probabilities = functional.softmax(
            (logits * self.sharpness + noise) / self.temperature, dim=0
        )
        values = (probabilities * self.candidates).sum(0)
        return (
            values.reshape(*scales.shape, 128) * scales.to(self.dtype)[..., None]
        ).flatten(1)

    def forward(self, inputs):
        weight = checkpoint(
            self.soft_weight, self.logits, self.scales, use_reentrant=False,
            preserve_rng_state=True,
        )
        return functional.linear(inputs, weight, self.bias)

    @torch.no_grad()
    def finalize(self):
        codes = self.logits.argmax(0)
        if self.bits > 2:
            codes = (self.initial_codes.long() + codes - 2).clamp(0, len(self.grid) - 1)
        return mk.Q(
            codes=codes.to(torch.uint8),
            params=self.quantized.params | {"scales": self.scales.half().float()},
            decode=self.quantized.decode,
            codec=self.quantized.codec,
            bits=self.bits * codes.numel() + 16 * self.scales.numel(),
            metadata=self.quantized.metadata | {"scale_fmt": "fp16", "trainable": ["scales"]},
        )


@torch.no_grad()
def lion_step(parameters, momentum, learning_rate, weight_decay=0.0):
    gradients = [parameter.grad for parameter in parameters]
    update = torch._foreach_mul(momentum, 0.9)
    torch._foreach_add_(update, gradients, alpha=0.1)
    torch._foreach_sign_(update)
    torch._foreach_mul_(parameters, 1 - learning_rate * weight_decay)
    torch._foreach_add_(parameters, update, alpha=-learning_rate)
    torch._foreach_mul_(momentum, 0.95)
    torch._foreach_add_(momentum, gradients, alpha=0.05)


def stage_output(block, call, stage):
    if stage in {"q_proj", "k_proj"}:
        return block.self_attn.get_submodule(stage)(block.input_layernorm(call.hidden().cuda()))
    if stage == "attention":
        hidden = block.input_layernorm(call.hidden().cuda())
        return extract_hidden(call.with_hidden(hidden).run(block.self_attn))
    return extract_hidden(call.run(block))


def train_stage(model, block, teacher, calls, names, stage, arguments):
    start = time.perf_counter()
    prefix = model.architecture.block_name(block.self_attn.layer_idx)
    originals = {name: block.get_submodule(name) for name in names}
    quantizers = {
        name: GSQLinear(original, model.quantized[f"{prefix}.{name}"].to("cuda"), arguments.bits)
        for name, original in originals.items()
    }
    for name, quantizer in quantizers.items():
        block.set_submodule(name, quantizer)
    logits = [quantizer.logits for quantizer in quantizers.values()]
    scales = [quantizer.scales for quantizer in quantizers.values()]
    logit_momentum = [torch.zeros_like(parameter) for parameter in logits]
    scale_momentum = [torch.zeros_like(parameter) for parameter in scales]
    steps = arguments.epochs * math.ceil(arguments.sequences / 64)
    step = 0
    for epoch in range(arguments.epochs):
        total_loss = torch.zeros((), device="cuda")
        for indices in torch.randperm(arguments.sequences).split(64):
            fraction = step / max(steps - 1, 1)
            for quantizer in quantizers.values():
                quantizer.temperature = 2 - 1.95 * fraction
                quantizer.sharpness = 100 + 400 * fraction
            for parameter in logits + scales:
                parameter.grad = None
            for selection in indices.split(arguments.microbatch_size):
                selected = selection.tolist()
                call = calls.batch(selected)
                with torch.no_grad():
                    target = stage_output(teacher, call, stage)
                prediction = stage_output(block, call, stage)
                loss = functional.mse_loss(prediction.float(), target.float())
                loss = loss * len(selected) / len(indices)
                loss.backward()
                total_loss += loss.detach()
            learning_rate = 0.1 + 0.45 * (1 + math.cos(math.pi * fraction))
            lion_step(logits, logit_momentum, 1e-4 * learning_rate, weight_decay=1.0)
            lion_step(scales, scale_momentum, 5e-5 * learning_rate)
            step += 1
            if step % max(1, math.ceil(arguments.sequences / 64) // 8) == 0:
                print(
                    f"\rBlock {block.self_attn.layer_idx:>2}  {stage:<9}  "
                    f"epoch {epoch + 1:>2}/{arguments.epochs}  step {step:>4}/{steps}",
                    end="", flush=True,
                )
        print(
            f"\rBlock {block.self_attn.layer_idx:>2}  {stage:<9}  "
            f"epoch {epoch + 1:>2}/{arguments.epochs}  "
            f"MSE {float(total_loss / math.ceil(arguments.sequences / 64)):.7g}",
            end="", flush=True,
        )
    print(flush=True)
    with torch.no_grad():
        for name, quantizer in quantizers.items():
            quantized = quantizer.finalize()
            original = originals[name]
            original.weight.copy_(quantized.w.to(original.weight.dtype))
            block.set_submodule(name, original)
            model.quantized[f"{prefix}.{name}"] = quantized.to("cpu", detach=True)
    return (time.perf_counter() - start) / len(names)


def tune_scales(model, teacher, batches, arguments):
    replacements = {}
    for name, representation in list(model.quantized.items()):
        original = model.module.get_submodule(name)
        quantized = representation.to("cuda")
        replacement = CodecLinear(original, quantized)
        replacements[name] = original.cpu(), replacement
        model.module.set_submodule(name, replacement)
        model.quantized[name] = quantized
    parameters = [
        parameter for _, replacement in replacements.values()
        for parameter in replacement.parameters_by_name.values()
    ]
    momentum = [torch.zeros_like(parameter) for parameter in parameters]
    teacher.cuda().eval()
    model.module.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.module.train()
    total_steps = math.ceil(len(batches) / 64)
    for step, indices in enumerate(torch.randperm(len(batches)).split(64)):
        for parameter in parameters:
            parameter.grad = None
        total_loss = torch.zeros((), device="cuda")
        for selection in indices.split(arguments.microbatch_size):
            selected = selection.tolist()
            tokens = torch.cat([batches[index]["input_ids"] for index in selected]).cuda()
            with torch.no_grad():
                target = teacher.model(tokens, use_cache=False).last_hidden_state.flatten(0, 1)
            hidden = model.module.model(tokens, use_cache=False).last_hidden_state
            flattened = hidden.detach().flatten(0, 1)
            gradient = torch.empty_like(flattened)
            for offset in range(0, len(flattened), 64):
                inputs = flattened[offset : offset + 64].requires_grad_(True)
                prediction = (
                    model.module.lm_head(inputs).float() / arguments.distillation_temperature
                )
                with torch.no_grad():
                    reference = teacher.lm_head(target[offset : offset + 64]).float()
                    reference = reference / arguments.distillation_temperature
                loss = functional.kl_div(
                    prediction.log_softmax(-1), reference.log_softmax(-1),
                    reduction="batchmean", log_target=True,
                ) * arguments.distillation_temperature**2
                loss = loss * len(inputs) / len(flattened) * len(selected) / len(indices)
                gradient[offset : offset + len(inputs)] = torch.autograd.grad(loss, inputs)[0]
                total_loss += loss.detach()
            hidden.backward(gradient.reshape_as(hidden))
        fraction = step / max(total_steps - 1, 1)
        lion_step(
            parameters, momentum,
            arguments.scale_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * fraction))),
        )
        print(
            f"\rScale distillation  {step + 1:>3}/{total_steps}  KL {float(total_loss):.7g}",
            end="", flush=True,
        )
    print(flush=True)
    model.module.gradient_checkpointing_disable()
    model.module.eval()
    with torch.no_grad():
        for name, (original, replacement) in replacements.items():
            for parameter in replacement.parameters_by_name.values():
                parameter.copy_(parameter.half().float())
            original.cuda().weight.copy_(replacement.weight.to(original.weight.dtype))
            model.module.set_submodule(name, original)
            model.quantized[name] = model.quantized[name].to("cpu", detach=True)
    teacher.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--bits", type=int, choices=(2, 3), default=2)
    parser.add_argument("--output", type=Path, default=Path("artifacts/llama_gsq"))
    parser.add_argument("--activation-cache-mib", type=int, default=512)
    parser.add_argument("--shuffle-buffer-size", type=int, default=1024)
    parser.add_argument("--sequences", type=int, default=4096)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--gptq-sequences", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--stability-samples", type=int, default=256)
    parser.add_argument("--scale-lr", type=float, default=5e-5)
    parser.add_argument("--distillation-temperature", type=float, default=1.0)
    parser.add_argument("--skip-scale-tuning", action="store_true")
    parser.add_argument("--tasks", default="arc_easy,arc_challenge,hellaswag,winogrande,piqa")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.manual_seed(arguments.seed)
    source = mk.load(arguments.model, dtype="bfloat16", attn_implementation="sdpa")
    source.module.requires_grad_(False)
    teacher = copy.deepcopy(source.module).cpu().eval()
    model = mk.QModel(source.module, source.tokenizer, name=source.name)
    corpus = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True
    ).shuffle(seed=arguments.seed, buffer_size=arguments.shuffle_buffer_size)
    batches = []
    buffer = []
    print("Preparing FineWeb-Edu calibration sequences.", flush=True)
    for documents in corpus.iter(batch_size=64):
        buffer.extend(itertools.chain.from_iterable(
            source.tokenizer(documents["text"], verbose=False)["input_ids"]
        ))
        while len(buffer) >= arguments.sequence_length:
            batches.append({"input_ids": torch.tensor(buffer[: arguments.sequence_length])[None]})
            del buffer[: arguments.sequence_length]
        print(
            f"\rPreparing FineWeb-Edu sequences  "
            f"{min(len(batches), arguments.sequences + arguments.gptq_sequences):>5}/"
            f"{arguments.sequences + arguments.gptq_sequences}",
            end="", flush=True,
        )
        if len(batches) >= arguments.sequences + arguments.gptq_sequences:
            break
    batches = batches[: arguments.sequences + arguments.gptq_sequences]
    print(flush=True)
    print("Capturing one GPU call template; activations will be computed on demand.", flush=True)
    session = CalibrationSession(
        model, batches[:1], sequential=True, sample_rows=arguments.stability_samples,
        cache_dir=None, need_targets=False, retain_history=False,
        storage="cuda", token_energy_limit=None,
    )
    session.prepare()
    templates = [session.block_calls(index)[0] for index in range(len(model.blocks))]
    initialization = mk.gptq(
        mk.int(
            arguments.bits, group=128, scale_fmt="fp32",
            scale=lambda values: values.abs().amax(-1).clamp_min(1e-12)
            * 2 / (2**arguments.bits - 1),
        ),
        refit=128, damp=0.01,
    )
    stages = (
        ("q_proj", ("self_attn.q_proj",)),
        ("k_proj", ("self_attn.k_proj",)),
        ("attention", ("self_attn.v_proj", "self_attn.o_proj")),
        ("block", ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")),
    )
    for index, block in enumerate(model.blocks):
        print(
            f"\nBlock {index + 1}/{len(model.blocks)}  |  staged GSQ-{arguments.bits}  |  "
            f"VRAM activation cache {arguments.activation_cache_mib} MiB",
            flush=True,
        )
        original = teacher.model.layers[index].cuda()
        calls = PrefixCalls(model, templates, index, batches, arguments.activation_cache_mib)
        statistics_session = copy.copy(session)
        statistics_session.calls = list(session.calls)
        statistics_session.calls[index] = calls[arguments.sequences :]
        statistics = BlockStatistics(
            statistics_session, original, index, model.architecture.block_name(index)
        )
        print(f"Collecting GPTQ statistics from {arguments.gptq_sequences} sequences.", flush=True)
        statistics.collect_many({"H": (None, "mean"), "X": (None, "sample")})
        calls.cache.clear()
        contexts = {}
        report_start = len(model.layer_reports)
        for name, projection in list(block.named_modules()):
            if not isinstance(projection, nn.Linear):
                continue
            full_name = f"{model.architecture.block_name(index)}.{name}"
            context = mk.Ctx(
                full_name, projection, block, index, seed=arguments.seed,
                provider=statistics.provider(full_name, projection.weight.device),
                device=projection.weight.device,
            )
            contexts[name] = context
            convert_layer(
                model, full_name, projection, initialization, context, retain_device=False
            )
        reports = {report.name: report for report in model.layer_reports[report_start:]}
        del model.layer_reports[report_start:]
        durations = {}
        for stage, names in stages:
            duration = train_stage(model, block, original, calls, names, stage, arguments)
            durations.update(dict.fromkeys(names, duration))
        monitor = conversion_monitor(model, stability=True, top_k=8, progress=True)
        with torch.no_grad():
            for name, context in contexts.items():
                inputs = context.X.to(model.dtype)
                reference = original.get_submodule(name)
                projection = block.get_submodule(name)
                measurement = mk.decision_stability(
                    functional.linear(inputs, reference.weight, reference.bias),
                    functional.linear(inputs, projection.weight, projection.bias),
                )
                model.layer_reports.append(replace(
                    reports[context.name], bits=model.quantized[context.name].bits,
                    loss=float(mk.proxy_loss(reference.weight, projection.weight, context)),
                    seconds=reports[context.name].seconds + durations[name],
                    method=f"GSQ-{arguments.bits}",
                ))
                monitor(context, measurement, model.layer_reports[-1].seconds)
        del calls, statistics_session, statistics, contexts, context
        original.cpu()
    del session, templates
    if arguments.bits == 2 and not arguments.skip_scale_tuning:
        tune_scales(model, teacher, batches[: arguments.sequences], arguments)
    model.save(arguments.output, overwrite=True)
    probe = mk.data("wikitext2", n=4, seq=256, split="test", tokenizer=model.tokenizer)
    teacher_model = mk.Model(teacher.cuda(), model.tokenizer)
    unchanged = 0
    tested = 0
    with torch.no_grad():
        for batch in probe:
            reference = teacher_model(**batch, use_cache=False).logits[:, :-1].argmax(-1)
            prediction = model(**batch, use_cache=False).logits[:, :-1].argmax(-1)
            unchanged += int((reference == prediction).sum())
            tested += reference.numel()
    del teacher_model, teacher, source
    torch.cuda.empty_cache()
    metrics = {
        "paper": "https://arxiv.org/abs/2604.18556v2",
        "reference_commit": "03fc16484c369e3127225615d5e03e8d3a6043e3",
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "scale_distillation": (
            "Teacher-to-student KL; temperature and optimizer lr are implementation choices."
        ),
        "stability_stage": "Before end-to-end scale tuning",
        "activation_strategy": "Bounded CUDA cache; GPU prefix and teacher recomputation",
        "layers": model.stability_reports,
        "next_token_dbs": unchanged / tested,
        "tested_next_token_decisions": tested,
        "quantized_bpw": model.bpw,
        "wikitext2_ppl": mk.ppl(
            model, data="wikitext2", budget="full", seq=arguments.sequence_length
        ),
    }
    if arguments.tasks:
        metrics["tasks"] = mk.eval(
            model, arguments.tasks.split(","), batch_size=1,
            max_length=4096, num_fewshot=0,
        )["results"]
    (arguments.output / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"\n{'Next-token DBS':<26} {metrics['next_token_dbs']:.2%}\n"
        f"{'WikiText-2 perplexity':<26} {metrics['wikitext2_ppl']:.4f}\n"
        f"{'Quantized bits/weight':<26} {metrics['quantized_bpw']:.4f}\n"
        f"{'Checkpoint':<26} {arguments.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
