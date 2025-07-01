from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


def ensure(requirements_file: str | Path | None = None) -> None:
    """Ensure listed packages can be imported, installing them if missing."""
    if requirements_file is None:
        requirements_file = (
            Path(__file__).resolve().parents[2] / "requirements.txt"
        )
    else:
        requirements_file = Path(requirements_file)

    try:
        packages = [
            line.strip().split("==")[0]
            for line in requirements_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    except Exception:
        return

    missing: list[str] = []
    for pkg in packages:
        module_name = pkg.replace("-", "_")
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(pkg)

    if missing:
        subprocess.call([sys.executable, "-m", "pip", "install", *missing])
