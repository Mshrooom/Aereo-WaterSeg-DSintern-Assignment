from __future__ import annotations

import argparse
from pathlib import Path

from aereo_water.inference.predictor import SegFormerPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selected-model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask-output", required=True)
    parser.add_argument("--overlay-output", required=True)
    parser.add_argument("--log-output", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()

    predictor = SegFormerPredictor(
        args.checkpoint,
        selected_model_path=args.selected_model,
        device=args.device,
        log_path=args.log_output,
        warmup_runs=1,
    )
    mask, _, metadata = predictor.predict(args.image)
    predictor.save_mask(mask, args.mask_output)
    predictor.save_overlay(args.image, mask, args.overlay_output)
    print(metadata)


if __name__ == "__main__":
    main()
