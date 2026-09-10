#!/usr/bin/env python3
"""在独立 Python 进程中依次运行四个正式 benchmark 配置。

进程隔离可以释放上一个数据集的 GPU 模型/图内存，也让单数据集失败易于重跑。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """验证配置存在，并以失败即停的方式调用 `python -m warp.run`。"""
    parser = argparse.ArgumentParser(description="Run all WARP-G paper configurations in isolated processes")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/paper"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--datasets", nargs="*", default=["hotpotqa", "2wiki", "musique", "popqa"])
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Forwarded to warp.run; yaml still defines the full fold split")
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
