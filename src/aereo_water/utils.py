from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_dumps(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def json_dump(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, indent=indent, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return output


def json_load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(data: Any) -> str:
    return sha256_text(canonical_json_dumps(data))


def sha256_dataframe(frame, *, columns: list[str] | None = None) -> str:
    selected = frame.loc[:, columns] if columns is not None else frame
    normalized = selected.copy()
    normalized = normalized.sort_values(
        list(normalized.columns),
        kind="mergesort",
    ).reset_index(drop=True)
    payload = normalized.to_csv(index=False, lineterminator="\n")
    return sha256_text(payload)


def get_git_commit(repository_dir: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repository_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    try:
        import torch

        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except ImportError:
        pass


def available_disk_gb(path: str | Path) -> float:
    usage = shutil.disk_usage(Path(path))
    return usage.free / (1024**3)


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    return sum(
        item.stat().st_size
        for item in root.rglob("*")
        if item.is_file()
    )


def ensure_minimum_disk(path: str | Path, minimum_gb: float) -> None:
    free_gb = available_disk_gb(path)
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"Only {free_gb:.2f} GB is free under {path}; "
            f"{minimum_gb:.2f} GB is required."
        )
