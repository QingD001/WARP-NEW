#!/usr/bin/env python3
"""Watch a warp.run process and restart after CUDA OOM with a smaller batch or lower dtype.

Overrides go through WARP_EMBEDDING_BATCH_SIZE / WARP_EMBEDDING_DTYPE so the paper
YAML and fold-checkpoint signature stay unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

OOM_MARKERS = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "torch.cuda.OutOfMemoryError",
    "CUDA error: out of memory",
    "Tried to allocate",
)
SUCCESS_MARKERS = (
    "quality_cost_curve",
    "reader_evaluation",
)


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return Path(f"/proc/{pid}").exists()


def _log_text(path: Path, start: int = 0) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[start:].decode("utf-8", errors="replace")


def _looks_oom(text: str, pid: int) -> bool:
    if any(marker in text for marker in OOM_MARKERS):
        return True
    try:
        dmesg = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=5)
        blob = dmesg.stdout + dmesg.stderr
    except (OSError, subprocess.TimeoutExpired):
        blob = ""
    return f"Killed process {pid}" in blob or (f"pid={pid}" in blob and "oom" in blob.lower())


def _looks_success(output: Path, text: str) -> bool:
    if output.is_file() and output.stat().st_size > 0:
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict) and "baselines" in payload:
            return True
    return all(marker in text for marker in SUCCESS_MARKERS)


def _next_settings(batch: int, dtype: str) -> tuple[int, str, str] | None:
    if batch > 1:
        return max(1, batch // 2), dtype, f"embedding_batch_size {batch}->{max(1, batch // 2)}"
    if dtype == "auto":
        return 1, "float16", "embedding_model_dtype auto->float16"
    if dtype == "float16":
        return 1, "bfloat16", "embedding_model_dtype float16->bfloat16"
    return None


def _kill_tree(pid: int) -> None:
    if not _alive(pid):
        return
    children = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True)
    for line in children.stdout.split():
        try:
            _kill_tree(int(line))
        except ValueError:
            continue
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + 15
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.5)
    if _alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _load_state(path: Path, defaults: dict) -> dict:
    if path.exists():
        return {**defaults, **json.loads(path.read_text(encoding="utf-8"))}
    return dict(defaults)


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _start_run(args: argparse.Namespace, batch: int, dtype: str) -> int:
    env = os.environ.copy()
    env["HF_HOME"] = env.get("HF_HOME") or "/root/data/hf-cache"
    env.pop("TRANSFORMERS_CACHE", None)
    env["OPENAI_BASE_URL"] = env.get("OPENAI_BASE_URL") or "https://yibuapi.com/v1"
    env["WARP_EMBEDDING_BATCH_SIZE"] = str(batch)
    env["WARP_EMBEDDING_DTYPE"] = dtype
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES") or "0"
    handle = args.log.open("a", encoding="utf-8")
    handle.write(
        f"\n==== oom-watch restart {time.strftime('%Y-%m-%dT%H:%M:%S')} "
        f"batch={batch} dtype={dtype} ====\n"
    )
    handle.flush()
    proc = subprocess.Popen(
        [args.python, "-u", "-m", "warp.run", "--config", str(args.config), "--output", str(args.output)],
        cwd=str(args.cwd),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    return proc.pid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path("/root/zjl/WARP-G"))
    parser.add_argument("--python", default="python3")
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--max-restarts", type=int, default=4)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    state_path = args.state or args.output.with_name("oom-watch.json")
    state = _load_state(state_path, {
        "pid": args.pid,
        "batch": 8,
        "dtype": "auto",
        "restarts": 0,
        "status": "watching",
    })
    pid = int(state["pid"])
    offset = args.log.stat().st_size if args.log.exists() else 0
    print(f"watch pid={pid} log={args.log} state={state_path}", flush=True)
    while True:
        time.sleep(args.interval)
        chunk = _log_text(args.log, offset)
        offset += len(chunk.encode("utf-8", errors="replace")) if chunk else 0
        if _alive(pid):
            if _looks_oom(chunk, pid):
                print(f"OOM in live process {pid}; stopping it", flush=True)
                _kill_tree(pid)
            else:
                continue
        full = _log_text(args.log)
        if _looks_success(args.output, full):
            state["status"] = "completed"
            _save_state(state_path, state)
            print("run finished", flush=True)
            return 0
        if not _looks_oom(full, pid) and not _looks_oom(chunk, pid):
            state["status"] = "exited_without_oom"
            _save_state(state_path, state)
            print(f"pid {pid} exited without OOM evidence; not restarting", flush=True)
            return 1
        if int(state["restarts"]) >= args.max_restarts:
            state["status"] = "oom_retries_exhausted"
            _save_state(state_path, state)
            print("max OOM restarts reached", flush=True)
            return 2
        nxt = _next_settings(int(state["batch"]), str(state["dtype"]))
        if nxt is None:
            state["status"] = "oom_settings_exhausted"
            _save_state(state_path, state)
            print("no smaller batch/dtype left", flush=True)
            return 2
        batch, dtype, reason = nxt
        print(f"restart after OOM: {reason}", flush=True)
        time.sleep(3)
        pid = _start_run(args, batch, dtype)
        state.update({"pid": pid, "batch": batch, "dtype": dtype, "restarts": int(state["restarts"]) + 1,
                      "status": "restarted", "reason": reason})
        _save_state(state_path, state)
        offset = args.log.stat().st_size if args.log.exists() else 0


if __name__ == "__main__":
    raise SystemExit(main())
