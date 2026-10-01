"""Run the synthetic microbenchmarks and write one result file.

    uv run python -m benchmarks.microbenchmarks
    uv run python -m benchmarks.microbenchmarks --filter "algorithms/gptq*" "kernels/*"
    uv run python -m benchmarks.microbenchmarks --cold-kernel-cache
"""

import argparse
import os
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--filter", nargs="+", metavar="PATTERN")
    parser.add_argument("--list", action="store_true", help="print case names and exit")
    parser.add_argument(
        "--cold-kernel-cache",
        action="store_true",
        help="compile every Triton kernel in this process, so first-call times include compilation",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    cache_directory = None
    if arguments.cold_kernel_cache:
        # Triton reads its cache location when it is first imported.
        cache_directory = tempfile.TemporaryDirectory(prefix="mlkit_triton_cache_")
        os.environ["TRITON_CACHE_DIR"] = cache_directory.name

    from benchmarks import harness
    from benchmarks.microbenchmarks import all_cases

    cases = harness.select(all_cases(), arguments.filter)
    if arguments.list:
        for case in cases:
            print(case.identifier)
        return
    records = harness.run(cases)
    output = arguments.output or harness.RESULTS_DIRECTORY / "microbenchmarks.json"
    harness.write_results(output, {
        "cold_kernel_cache": arguments.cold_kernel_cache,
        "measurements": records,
    })
    print(f"wrote {len(records)} measurements to {output}")


if __name__ == "__main__":
    main()
