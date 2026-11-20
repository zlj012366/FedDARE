from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    python_files = sorted((ROOT / "src").rglob("*.py"))
    python_files += sorted((ROOT / "scripts").rglob("*.py"))
    python_files.append(ROOT / "main.py")
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    configs = sorted((ROOT / "configs").rglob("*.yaml"))
    from feddare.config import load_config

    for path in configs:
        if not isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict):
            raise TypeError(f"{path} has no mapping at its root")
        load_config(str(path))
    required = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "pyproject.toml",
        ROOT / "docs" / "IMPLEMENTATION.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package_init = (ROOT / "src" / "feddare" / "__init__.py").read_text(encoding="utf-8")
    project_version = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    package_version = re.search(r'^__version__ = "([^"]+)"$', package_init, re.MULTILINE)
    if not project_version or not package_version:
        raise ValueError("Version metadata is missing")
    if project_version.group(1) != package_version.group(1):
        raise ValueError("Project and package versions do not match")
    if project_version.group(1) != "0.1.0":
        raise ValueError("Expected version 0.1.0")
    repository_text = (ROOT / "README.md").read_text(encoding="utf-8")
    if "<repository-url>" in repository_text or "[repository URL]" in repository_text:
        raise ValueError("README contains a repository URL placeholder")
    print(f"Validated {len(python_files)} Python files and {len(configs)} configs.")


if __name__ == "__main__":
    main()
