"""Collect an exact number of custom AirWrite samples for one label."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = frozenset("abcdefghijklmnopqrstuvwxyz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect exactly LIMIT new samples for one AirWrite character."
    )
    parser.add_argument("label", help="One ASCII letter (for example: r)")
    parser.add_argument("--limit", type=int, required=True, help="Number of new images to collect")
    parser.add_argument(
        "--style",
        choices=("uppercase", "lowercase"),
        help="Writing style (default: inferred from LABEL case)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw_airwrite",
        help="Dataset root (default: data/raw_airwrite)",
    )
    parser.add_argument(
        "--save-key",
        default="v",
        help="App key used to save the current canvas (default: v)",
    )
    return parser.parse_args()


def normalize_label(value: str, style: str) -> str:
    label = value.strip()
    if len(label) != 1 or label.lower() not in LABELS:
        raise ValueError("label must be exactly one ASCII letter")
    return label.lower() if style == "lowercase" else label.upper()


def main() -> int:
    arguments = parse_args()
    if arguments.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if len(arguments.save_key) != 1:
        raise ValueError("--save-key must contain exactly one character")

    style = arguments.style or ("uppercase" if arguments.label.isupper() else "lowercase")
    label = normalize_label(arguments.label, style)
    environment = os.environ.copy()
    environment.update(
        {
            "ENABLE_DATASET_CAPTURE": "true",
            "DATASET_CAPTURE_OUTPUT_DIR": str(arguments.output_dir),
            "DATASET_CAPTURE_WRITING_STYLE": style,
            "DATASET_CAPTURE_INITIAL_LABEL": label,
            "DATASET_CAPTURE_SAVE_KEY": arguments.save_key,
            "DATASET_CAPTURE_CLEAR_AFTER_SAVE": "true",
            "DATASET_CAPTURE_LIMIT": str(arguments.limit),
        }
    )

    print(f"Collecting {arguments.limit} new {label!r} samples.")
    print(f"Press {arguments.save_key!r} after each drawing; press q or Esc to cancel.")
    print(f"Progress: 0/{arguments.limit}")
    completed = subprocess.run(
        [sys.executable, "-m", "app.main"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
