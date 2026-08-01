from __future__ import annotations

import argparse
from pathlib import Path

from aereo_water.config import load_config
from aereo_water.data.manifest import (
    assert_split_integrity,
    assign_exact_split,
    discover_pairs,
    make_portable_registry,
    near_duplicate_audit,
    recover_historical_split,
    validate_manifest,
    write_data_registry,
)
from aereo_water.utils import get_git_commit, json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--historical-split-csv")
    parser.add_argument("--repository", default=".")
    parser.add_argument(
        "--skip-near-duplicate-audit",
        action="store_true",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(args.images, args.masks)
    validated, errors = validate_manifest(pairs)
    errors.to_csv(output / "data_errors.csv", index=False)
    if not errors.empty:
        raise RuntimeError(
            f"{len(errors)} invalid pairs were found. "
            f"Inspect {output / 'data_errors.csv'}."
        )

    if args.historical_split_csv:
        manifest = recover_historical_split(
            validated,
            args.historical_split_csv,
        )
    else:
        manifest = assign_exact_split(
            validated,
            train_count=config.data.train_count,
            validation_count=config.data.validation_count,
            test_count=config.data.test_count,
            seed=config.data.split_seed,
        )
    assert_split_integrity(manifest)
    manifest.to_csv(output / "runtime_manifest.csv", index=False)
    portable = make_portable_registry(
        manifest,
        dataset_root=Path(args.images).parent,
    )
    write_data_registry(
        portable,
        output_csv=output / "split_registry.csv",
        output_json=output / "data_registry.json",
        dataset_name="Satellite Images of Water Bodies",
        dataset_source="Kaggle",
        split_seed=config.data.split_seed,
        git_commit=get_git_commit(args.repository),
        duplicate_policy=(
            "SHA-256 exact rejection and complete perceptual-hash audit"
        ),
    )
    if not args.skip_near_duplicate_audit:
        audit = near_duplicate_audit(
            manifest,
            hamming_threshold=(
                config.data.near_duplicate_hamming_threshold
            ),
            maximum_pairs=None,
        )
        audit.to_csv(output / "near_duplicate_audit.csv", index=False)
        metadata = dict(audit.attrs)
        metadata["suspected_cross_split_pairs"] = int(len(audit))
        json_dump(
            metadata,
            output / "near_duplicate_audit_metadata.json",
        )


if __name__ == "__main__":
    main()
