"""Run activation-aware, lattice, trellis and fine-tuning recipes on a causal model."""

import argparse

import mlkit as mk


def main() -> None:
    recipes = {
        "gptq": mk.gptq(mk.int(4, group=128)),
        "shared-awq": mk.awq(mk.int(4, group=128), shared=True),
        "lattice": mk.incoherent(
            mk.ldlq(mk.scaled(mk.grid.e8p(), group=None), step=8, refit=None)
        ),
        "trellis": mk.incoherent(mk.ldlq(mk.trellis(L=8), step=16, refit=None)),
        "w4a4": mk.Recipe(
            weights=mk.gptq(mk.int(4, group=None)),
            acts=mk.int(4, group=None),
            kv=mk.int(4, group=64),
            transforms=[mk.rotate("hadamard")],
        ),
        "finetune": mk.Recipe(
            weights=mk.incoherent(
                mk.gptq(mk.int(4, group=128)), train_signs=True
            ),
            passes=[mk.finetune(steps=5, bs=2)],
        ),
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    parser.add_argument("--method", choices=recipes, default="gptq")
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--evaluation-batches", type=int, default=4)
    arguments = parser.parse_args()
    model = mk.load(arguments.model, dtype="float16")
    calibration = mk.data(
        "wikitext2", n=arguments.calibration_batches, seq=arguments.sequence, seed=17
    )
    evaluation = mk.data(
        "wikitext2", n=arguments.evaluation_batches, seq=arguments.sequence, split="test"
    )
    recipe = recipes[arguments.method]
    if isinstance(recipe, mk.Recipe):
        recipe = recipe.replace(name=arguments.method)
    else:
        recipe = mk.Recipe(weights=recipe, name=arguments.method)
    mk.compare(
        model,
        [recipe],
        calib=calibration,
        data=evaluation,
        budget="full",
        seq=arguments.sequence,
    )


if __name__ == "__main__":
    main()
