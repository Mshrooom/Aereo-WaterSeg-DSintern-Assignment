from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ComputeBudget:
    hpo_optimizer_steps: int
    confirmation_optimizer_steps: int
    stability_optimizer_steps: int
    final_optimizer_steps: int
    estimated_total_optimizer_steps: int
    estimated_training_runs: int
    minimum_free_disk_gb: float

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_compute_budget(
    *,
    hpo_trials: int,
    hpo_epochs: int,
    hpo_train_images: int,
    confirmation_runs: int,
    confirmation_epochs: int,
    stability_runs: int,
    stability_epochs: int,
    full_train_images: int,
    final_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    minimum_free_disk_gb: float = 20.0,
) -> ComputeBudget:
    effective_batch = batch_size * gradient_accumulation_steps

    def steps(images: int, epochs: int, runs: int) -> int:
        return math.ceil(images / effective_batch) * epochs * runs

    hpo_steps = steps(hpo_train_images, hpo_epochs, hpo_trials)
    confirmation_steps = steps(
        full_train_images,
        confirmation_epochs,
        confirmation_runs,
    )
    stability_steps = steps(
        full_train_images,
        stability_epochs,
        stability_runs,
    )
    final_steps = steps(full_train_images, final_epochs, 1)
    return ComputeBudget(
        hpo_optimizer_steps=hpo_steps,
        confirmation_optimizer_steps=confirmation_steps,
        stability_optimizer_steps=stability_steps,
        final_optimizer_steps=final_steps,
        estimated_total_optimizer_steps=(
            hpo_steps
            + confirmation_steps
            + stability_steps
            + final_steps
        ),
        estimated_training_runs=(
            hpo_trials + confirmation_runs + stability_runs + 1
        ),
        minimum_free_disk_gb=float(minimum_free_disk_gb),
    )
