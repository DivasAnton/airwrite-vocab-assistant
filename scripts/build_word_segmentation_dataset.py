import argparse
import csv
from pathlib import Path

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manifest for whole-word audit images")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/word_segmentation_eval/manifest.csv"),
    )
    return parser.parse_args()


def expected_word(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) > 1:
        return relative.parts[0]
    return path.stem.split("_")[0]


def main() -> None:
    args = parse_args()
    paths = sorted(
        path
        for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=("image_path", "expected_word"))
        writer.writeheader()
        for path in paths:
            writer.writerow(
                {
                    "image_path": path.as_posix(),
                    "expected_word": expected_word(path, args.input_dir),
                }
            )
    print(f"Wrote {len(paths)} rows to {args.output}")


if __name__ == "__main__":
    main()
