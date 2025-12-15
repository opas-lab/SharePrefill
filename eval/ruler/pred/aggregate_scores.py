# Copyright 2025 OPPO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Aggregate SharePrefill RULER scores across sequence lengths.

Given a results root directory that contains subdirectories named after
sequence lengths (e.g. 4096, 8192, ...), each with a `pred/summary.csv`
produced by `eval/ruler/eval/evaluate.py`, this script computes:

1. The average score per sequence length.
2. The overall average across all sequence lengths.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate RULER summary scores.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path.cwd(),
        help=(
            "Root directory containing length folders "
            "(default: current working directory)."
        ),
    )
    parser.add_argument(
        "--print-tasks",
        action="store_true",
        help="Also print the task-level scores for each length.",
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help=(
            "Write per-length and overall averages into <MODEL>_summary.csv "
            "alongside the results root."
        ),
    )
    return parser.parse_args()


def read_scores(summary_path: Path) -> Tuple[float, Dict[str, float]]:
    """Read a summary.csv and return (avg_score, per_task_scores)."""
    with summary_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() == "score":
                scores = {
                    f"task_{idx}": float(value)
                    for idx, value in enumerate(row[1:], start=1)
                    if value.strip()
                }
                if not scores:
                    raise ValueError(f"No score entries found in {summary_path}")
                avg_score = statistics.fmean(scores.values())
                return avg_score, scores
    raise ValueError(f"Unable to locate a 'Score' row in {summary_path}")


def find_summary_files(results_root: Path) -> Dict[str, Path]:
    """Locate summary.csv files keyed by their length directory name."""
    summaries: Dict[str, Path] = {}
    for length_dir in sorted(results_root.iterdir()):
        if not length_dir.is_dir():
            continue
        summary_path = length_dir / "pred" / "summary.csv"
        if summary_path.exists():
            summaries[length_dir.name] = summary_path
    if not summaries:
        raise FileNotFoundError(f"No summary.csv files found under {results_root}")
    return summaries


def main() -> None:
    args = parse_args()
    summaries = find_summary_files(args.results_root)

    per_length_avg: Dict[str, float] = {}

    print("Sequence Length,Average Score")
    for length, summary_path in summaries.items():
        avg_score, per_task = read_scores(summary_path)
        per_length_avg[length] = avg_score
        print(f"{length},{avg_score:.4f}")
        if args.print_tasks:
            for task_name, score in per_task.items():
                print(f"  {task_name}: {score:.4f}")

    overall = statistics.fmean(per_length_avg.values())
    print(f"\nOverall Average,{overall:.4f}")

    if args.write_summary:
        benchmark_name = args.results_root.name
        variant_name = args.results_root.parent.name
        model_name = args.results_root.parent.parent.name
        ordered_lengths = [
            length
            for length in sorted(
                per_length_avg,
                key=lambda length_key: int(length_key),
            )
        ]
        header = ["Model"] + ordered_lengths + ["Overall"]
        score_columns: list[str] = []
        for length in ordered_lengths:
            score_columns.append(f"{per_length_avg[length]:.4f}")
        row = [model_name, *score_columns, f"{overall:.4f}"]

        summary_path = args.results_root / "summary.csv"

        with summary_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerow(row)
        summary_label = " ".join([model_name, variant_name, benchmark_name])
        message = "\nWrote aggregated " f"{summary_label} summary to {summary_path}"
        print(message)


if __name__ == "__main__":
    main()
