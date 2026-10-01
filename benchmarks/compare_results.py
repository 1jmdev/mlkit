"""Compare two microbenchmark result files and report the ratio of every shared case.

    uv run python -m benchmarks.compare_results baseline.json candidate.json
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = ["median_ms", "minimum_ms", "first_call_ms", "additional_peak_memory_bytes"]


def load_measurements(path: Path) -> dict[str, dict[str, Any]]:
    document = json.loads(path.read_text())
    return {record["case"]: record for record in document["measurements"]}


def verdict(ratio: float, threshold: float) -> str:
    if ratio >= threshold:
        return "  improved"
    if ratio <= 1 / threshold:
        return "  REGRESSED"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--metric", default="median_ms", choices=METRICS)
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.05,
        help="ratio beyond which a case is reported as improved or regressed",
    )
    arguments = parser.parse_args()
    baseline = load_measurements(arguments.baseline)
    candidate = load_measurements(arguments.candidate)
    ratios = []
    print(f"{'case':58} {'baseline':>14} {'candidate':>14} {'ratio':>8}")
    for case in baseline:
        if case not in candidate:
            continue
        before = baseline[case][arguments.metric]
        after = candidate[case][arguments.metric]
        ratio = before / after if after else float("inf")
        ratios.append(ratio)
        print(
            f"{case:58} {before:14.3f} {after:14.3f} {ratio:7.2f}x"
            f"{verdict(ratio, arguments.threshold)}"
        )
    finite = [ratio for ratio in ratios if 0 < ratio < float("inf")]
    if finite:
        mean = statistics.geometric_mean(finite)
        print(f"geometric mean ratio over {len(finite)} cases: {mean:.2f}x")
    for case in sorted(set(baseline) ^ set(candidate)):
        owner = "baseline" if case in baseline else "candidate"
        print(f"only in {owner}: {case}")


if __name__ == "__main__":
    main()
