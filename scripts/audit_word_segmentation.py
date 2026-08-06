import argparse
import json
from pathlib import Path

import cv2

from app.word_recognition.hybrid_word_segmenter import HybridWordSegmenter
from app.word_recognition.word_input_snapshot import WordInputSnapshot
from app.word_recognition.word_writing_region import WordWritingRegion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit isolated-letter word segmentation")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/word_segmentation_audit")
    )
    parser.add_argument("--min-gap", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    height, width = image.shape[:2]
    result = HybridWordSegmenter(min_separator_gap=args.min_gap).segment(
        WordInputSnapshot(
            snapshot_id=args.image.stem,
            canvas_image=image,
            strokes=(),
            writing_region=WordWritingRegion(0, 0, width, height),
            completed_at_ms=0,
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated = image.copy()
    for segment in result.segments:
        box = segment.bounding_box
        cv2.rectangle(
            annotated,
            (box.x, box.y),
            (box.right - 1, box.bottom - 1),
            (0, 255, 0),
            2,
        )
    image_path = args.output_dir / f"{args.image.stem}_segments.png"
    report_path = args.output_dir / f"{args.image.stem}_segments.json"
    cv2.imwrite(str(image_path), annotated)
    report = {
        "image": str(args.image),
        "status": result.status.value,
        "segment_count": len(result.segments),
        "elapsed_ms": result.elapsed_ms,
        "segments": [
            {
                "position": segment.position,
                "segment_id": segment.segment_id,
                "bounding_box": {
                    "x": segment.bounding_box.x,
                    "y": segment.bounding_box.y,
                    "width": segment.bounding_box.width,
                    "height": segment.bounding_box.height,
                },
                "source_stroke_ids": segment.source_stroke_ids,
                "confidence": segment.segmentation_confidence,
            }
            for segment in result.segments
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
