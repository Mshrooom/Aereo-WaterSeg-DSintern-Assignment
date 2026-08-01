from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aereo_water.utils import json_dump, json_load, utc_now_iso


STAGE_ORDER = [
    "data",
    "hpo",
    "confirmation",
    "stability",
    "final_train",
    "calibrate",
    "evaluate",
    "inference",
    "api_test",
    "export",
]


@dataclass
class StageRecord:
    stage: str
    status: str
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    evidence: list[str] | None = None
    message: str | None = None


class StageState:
    """Persist and validate stage completion for resumable notebook runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            self.payload = json_load(self.path)
        else:
            self.payload = {
                "schema_version": 1,
                "stages": {},
                "updated_at_utc": utc_now_iso(),
            }

    def _save(self) -> None:
        self.payload["updated_at_utc"] = utc_now_iso()
        json_dump(self.payload, self.path)

    def start(self, stage: str) -> None:
        self._validate_stage(stage)
        self.payload["stages"][stage] = {
            "status": "running",
            "started_at_utc": utc_now_iso(),
            "completed_at_utc": None,
            "evidence": [],
            "message": None,
        }
        self._save()

    def complete(
        self,
        stage: str,
        *,
        evidence: list[str] | None = None,
        message: str | None = None,
    ) -> None:
        self._validate_stage(stage)
        record = self.payload["stages"].setdefault(stage, {})
        record.update(
            {
                "status": "complete",
                "completed_at_utc": utc_now_iso(),
                "evidence": evidence or [],
                "message": message,
            }
        )
        self._save()

    def fail(self, stage: str, message: str) -> None:
        self._validate_stage(stage)
        record = self.payload["stages"].setdefault(stage, {})
        record.update(
            {
                "status": "failed",
                "completed_at_utc": utc_now_iso(),
                "message": message,
            }
        )
        self._save()


    def invalidate_from(self, stage: str) -> None:
        """Remove a stage and every downstream completion record."""
        self._validate_stage(stage)
        position = STAGE_ORDER.index(stage)
        for affected in STAGE_ORDER[position:]:
            self.payload.get("stages", {}).pop(affected, None)
        self._save()

    def is_complete(self, stage: str) -> bool:
        return (
            self.payload.get("stages", {})
            .get(stage, {})
            .get("status")
            == "complete"
        )

    def require(self, stage: str) -> None:
        self._validate_stage(stage)
        position = STAGE_ORDER.index(stage)
        for predecessor in STAGE_ORDER[:position]:
            if predecessor in {"stability", "api_test"}:
                continue
            if not self.is_complete(predecessor):
                raise RuntimeError(
                    f"Stage '{stage}' requires completed stage "
                    f"'{predecessor}'."
                )

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if stage not in STAGE_ORDER:
            raise ValueError(
                f"Unknown stage '{stage}'. Expected one of {STAGE_ORDER}."
            )
