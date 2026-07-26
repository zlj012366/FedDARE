from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from typing import Any, Optional, Sequence

import yaml

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FedDARE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a YAML field; may be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def apply_override(config: Any, expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must look like section.key=value: {expression}")
    path, raw_value = expression.split("=", 1)
    parts = path.split(".")
    if len(parts) != 2:
        raise ValueError(f"Override must address one section and one field: {path}")
    section_name, field_name = parts
    if not hasattr(config, section_name):
        raise ValueError(f"Unknown config section: {section_name}")
    section = getattr(config, section_name)
    if not hasattr(section, field_name):
        raise ValueError(f"Unknown config field: {path}")
    setattr(section, field_name, yaml.safe_load(raw_value))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.device:
        config.runtime.device = args.device
    if args.rounds is not None:
        config.federation.rounds = args.rounds
    if args.seed is not None:
        config.runtime.seed = args.seed
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    for expression in args.set:
        apply_override(config, expression)
    config.validate()
    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
        return
    from .federation import FedDARERunner

    records = FedDARERunner(config).run()
    result = {
        key: (None if isinstance(value, float) and math.isnan(value) else value)
        for key, value in asdict(records[-1]).items()
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

