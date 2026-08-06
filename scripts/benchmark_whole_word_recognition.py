import argparse
from pathlib import Path

import numpy as np

from app.word_recognition.word_case_policy import WordCasePolicy
from scripts.test_whole_word_recognition import build_strategy, recognize_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark saved whole-word canvas recognition")
    parser.add_argument("image", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--case-policy", default="lowercase")
    args = parser.parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    strategy = build_strategy()
    policy = WordCasePolicy.from_string(args.case_policy)
    timings = [recognize_image(strategy, args.image, policy)[1] for _ in range(args.iterations)]
    print(
        f"runs={len(timings)} mean_ms={np.mean(timings):.2f} "
        f"p50_ms={np.percentile(timings, 50):.2f} "
        f"p95_ms={np.percentile(timings, 95):.2f}"
    )


if __name__ == "__main__":
    main()
