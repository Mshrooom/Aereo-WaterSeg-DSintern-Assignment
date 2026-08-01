from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from aereo_water.utils import canonical_json_dumps, utc_now_iso


FAILURE_COLUMNS = [
    "stage",
    "timestamp_utc",
    "configuration_json",
    "error_type",
    "error_message",
    "root_cause",
    "resolution",
    "rerun_status",
]


def append_failure(
    path: str | Path,
    *,
    stage: str,
    error: BaseException | None = None,
    configuration: dict[str, Any] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    root_cause: str = "",
    resolution: str = "",
    rerun_status: str = "pending",
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "stage": stage,
        "timestamp_utc": utc_now_iso(),
        "configuration_json": canonical_json_dumps(configuration or {}),
        "error_type": error_type or (type(error).__name__ if error else ""),
        "error_message": error_message or (str(error) if error else ""),
        "root_cause": root_cause,
        "resolution": resolution,
        "rerun_status": rerun_status,
    }
    exists = output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return output
