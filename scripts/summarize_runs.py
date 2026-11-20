from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def last_measured_row(path: Path):
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    measured = [row for row in rows if not math.isnan(float(row["test_accuracy"]))]
    if not measured:
        raise RuntimeError(f"No evaluated round in {path}")
    return measured[-1]


def mean_std(values):
    if len(values) == 1:
        return statistics.mean(values), 0.0
    return statistics.mean(values), statistics.stdev(values)


def report(name, values):
    if not values:
        return
    mean, standard_deviation = mean_std(values)
    print(f"{name}: {mean:.4f} +/- {standard_deviation:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize final FedDARE metrics.")
    parser.add_argument("run_root")
    args = parser.parse_args()
    files = sorted(Path(args.run_root).glob("seed_*/metrics.csv"))
    if not files:
        raise SystemExit(f"No seed_*/metrics.csv below {args.run_root}")
    rows = [last_measured_row(path) for path in files]
    print(f"runs: {len(rows)}")
    report("TAcc", [float(row["test_accuracy"]) for row in rows])
    report(
        "ASR",
        [
            float(row["attack_success_rate"])
            for row in rows
            if not math.isnan(float(row["attack_success_rate"]))
        ],
    )
    report(
        "Detection",
        [
            float(row["detection_rate"])
            for row in rows
            if "detection_rate" in row
            and not math.isnan(float(row["detection_rate"]))
        ],
    )
    report(
        "False positive",
        [
            float(row["false_positive_rate"])
            for row in rows
            if "false_positive_rate" in row
            and not math.isnan(float(row["false_positive_rate"]))
        ],
    )


if __name__ == "__main__":
    main()
