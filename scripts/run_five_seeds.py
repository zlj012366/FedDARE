from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one config with five seeds.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="./runs/five_seeds")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seeds", type=int, nargs=5, default=[2026, 2027, 2028, 2029, 2030]
    )
    args = parser.parse_args()
    for seed in args.seeds:
        command = [
            sys.executable,
            "-m",
            "feddare.cli",
            "--config",
            args.config,
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--output-dir",
            str(Path(args.output_root) / f"seed_{seed}"),
        ]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

