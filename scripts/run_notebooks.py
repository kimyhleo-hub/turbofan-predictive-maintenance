"""Execute project notebooks without overwriting the source notebooks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


NOTEBOOKS = [
    "01_eda.ipynb",
    "02_preprocessing.ipynb",
    "03_rul_model.ipynb",
    "04_optimization.ipynb",
    "05_evaluation.ipynb",
]


def repo_root() -> Path:
    here = Path(__file__).resolve()
    root = here.parents[1]
    if not (root / "notebooks").exists():
        raise RuntimeError(f"Could not locate notebooks directory from {here}")
    return root


def run_notebook(root: Path, notebook: str, output_dir: Path, timeout: int) -> None:
    source = root / "notebooks" / notebook
    output_name = f"{source.stem}.executed.ipynb"
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(source),
        "--output-dir",
        str(output_dir),
        "--output",
        output_name,
        f"--ExecutePreprocessor.timeout={timeout}",
        "--ExecutePreprocessor.kernel_name=python3",
    ]
    print(f"\n==> {notebook}", flush=True)
    subprocess.run(cmd, cwd=root, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="runs/notebooks",
        help="Directory for executed notebook copies.",
    )
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    root = repo_root()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for notebook in NOTEBOOKS:
        run_notebook(root, notebook, output_dir, args.timeout)

    print(f"\nExecuted notebooks written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
