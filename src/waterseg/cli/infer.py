from __future__ import annotations

import argparse
import json
from pathlib import Path

from waterseg.inference import WaterSegmenter, postprocess_mask, read_image_with_profile, save_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM water segmentation inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--points", default="", help='JSON, e.g. "[[120,80],[300,200]]"')
    parser.add_argument("--labels", default="", help='JSON, e.g. "[1,0]"')
    parser.add_argument("--box", default="", help='JSON, e.g. "[10,20,500,400]"')
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-component-area", type=int, default=0)
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--disable-tiling", action="store_true")
    args = parser.parse_args()

    image, profile = read_image_with_profile(args.image)
    segmenter = WaterSegmenter(args.checkpoint)
    points = json.loads(args.points) if args.points else None
    labels = json.loads(args.labels) if args.labels else None
    box = json.loads(args.box) if args.box else None
    if not args.disable_tiling and points is None and box is None:
        mask, _ = segmenter.segment_large_image(
            image, tile_size=args.tile_size, overlap=args.tile_overlap, threshold=args.threshold
        )
    else:
        mask, _ = segmenter.segment(image, points, labels, box, args.threshold)
    mask = postprocess_mask(mask, args.min_component_area, args.fill_holes)
    save_mask(args.output, mask, profile)
    print(f"Saved mask to {args.output}")


if __name__ == "__main__":
    main()
