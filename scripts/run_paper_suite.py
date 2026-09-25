#!/usr/bin/env python3
"""Run paper benchmark configs in isolated processes.

Process isolation releases GPU / graph memory after each dataset and makes a
single-dataset failure easy to rerun.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Require each config to exist, then call `python -m warp.run` and stop on failure."""
    parser = argparse.ArgumentParser(description="Run all WARP-G paper configurations in isolated processes")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/paper"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--datasets", nargs="*", default=["hotpotqa", "2wiki", "musique", "nq"])
    parser.add_argument("--max-folds", type=int, default=1,
                        help="Forwarded to warp.run (paper commands use 1)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        config = args.config_dir / f"{dataset}.yaml"
        if not config.exists():
            raise FileNotFoundError(config)
        output = args.output_dir / f"{dataset}.json"
        command = [
            sys.executable, "-m", "warp.run", "--config", str(config), "--output", str(output),
        ]
        if args.max_folds is not None:
            command.extend(["--max-folds", str(args.max_folds)])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
