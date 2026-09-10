"""Append-only raw experiment records; no model calls and no credential logging."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import threading
import time

_state = ContextVar("warp_audit", default={})
_lock = threading.Lock()


@contextmanager
def audit_scope(**labels):
    token = _state.set({**_state.get(), **labels})
    try:
        yield
    finally:
        _state.reset(token)


def audit_context():
    return dict(_state.get())


def emit(event, payload, context=None):
    state = dict(_state.get() if context is None else context)
    path = state.pop("path", None)
    if path is None:
        return
    def encode(value):
        if is_dataclass(value):
            return asdict(value)
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(f"Unsupported audit value: {type(value).__name__}")
    record = {"time_unix": time.time(), **state, "event": event, "payload": payload}
    rendered = json.dumps(record, ensure_ascii=False, default=encode, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
        handle.flush()
