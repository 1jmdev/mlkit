"""Run optional language-model task evaluation with an explicit development budget."""

import argparse
import json
from pathlib import Path

import mlkit as mk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    parser.add_argument("--tasks", nargs="+", default=["arc_easy", "piqa"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--recipe")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    model = mk.load(arguments.model, dtype="float16")
    if arguments.recipe is not None:
        calibration = mk.data("wikitext2", n=4, seq=128, seed=17)
        model = mk.quantize(model, arguments.recipe, calib=calibration)
    scores = mk.eval(
        model,
        tasks=arguments.tasks,
        limit=arguments.limit,
        batch_size=arguments.batch_size,
        bootstrap_iters=100,
        log_samples=False,
    )
    document = {
        "model": arguments.model,
        "recipe": arguments.recipe,
        "limit": arguments.limit,
        "results": scores["results"],
    }
    serialized = json.dumps(document, indent=2) + "\n"
    print(serialized, end="")
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized)


if __name__ == "__main__":
    main()
