from __future__ import annotations

import argparse

from waterseg.hybrid_inference import HybridWaterSegmenter
from waterseg.inference import postprocess_mask, read_image_with_profile, save_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic SegFormer-to-SAM water segmentation")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--segformer-checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--min-component-area", type=int, default=0)
    parser.add_argument("--fill-holes", action="store_true")
    args = parser.parse_args()

    image, profile = read_image_with_profile(args.image)
    segmenter = HybridWaterSegmenter(args.sam_checkpoint, args.segformer_checkpoint)
    mask, _ = segmenter.segment_large_image(image, args.tile_size, args.tile_overlap, args.threshold)
    mask = postprocess_mask(mask, args.min_component_area, args.fill_holes)
    save_mask(args.output, mask, profile)
    print(f"Saved automatic hybrid mask to {args.output}")


if __name__ == "__main__":
    main()
