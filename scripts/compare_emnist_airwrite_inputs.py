import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

from app.utils.config import preprocessing_alignment_settings

COMPARISON_METRICS = (
    ("foreground_ratio", "median"),
    ("bbox_width_ratio", "median"),
    ("bbox_height_ratio", "median"),
    ("centroid_x", "median"),
    ("centroid_y", "median"),
    ("mean_foreground_intensity", "mean"),
    ("component_count", "median"),
)


def read_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing audit report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"Preprocessing audit report must contain a JSON object: {path}")
    return cast(dict[str, Any], report)


def metric_value(report: dict[str, Any], metric: str, statistic: str) -> float:
    try:
        return float(report["metrics"][metric][statistic])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Report is missing metrics.{metric}.{statistic}") from error


def build_comparison(emnist: dict[str, Any], airwrite: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, dict[str, object]] = {}
    metric_pairs: dict[str, tuple[float, float]] = {}
    for metric, statistic in COMPARISON_METRICS:
        emnist_value = metric_value(emnist, metric, statistic)
        airwrite_value = metric_value(airwrite, metric, statistic)
        metric_pairs[metric] = (emnist_value, airwrite_value)
        metrics[metric] = {
            "statistic": statistic,
            "emnist": emnist_value,
            "airwrite": airwrite_value,
            "absolute_difference": abs(airwrite_value - emnist_value),
        }

    warnings = []
    emnist_width, airwrite_width = metric_pairs["bbox_width_ratio"]
    emnist_height, airwrite_height = metric_pairs["bbox_height_ratio"]
    if airwrite_width > emnist_width + 0.20 or airwrite_height > emnist_height + 0.20:
        warnings.append("AirWrite content appears substantially larger than EMNIST")
    if airwrite_width < emnist_width - 0.20 or airwrite_height < emnist_height - 0.20:
        warnings.append("AirWrite content appears substantially smaller than EMNIST")
    emnist_foreground, airwrite_foreground = metric_pairs["foreground_ratio"]
    if emnist_foreground > 0.0 and airwrite_foreground / emnist_foreground > 1.75:
        warnings.append("AirWrite foreground ratio is substantially denser than EMNIST")
    if airwrite_foreground > 0.0 and emnist_foreground / airwrite_foreground > 1.75:
        warnings.append("AirWrite foreground ratio is substantially sparser than EMNIST")
    return {
        "emnist_sample_count": emnist.get("sample_count"),
        "airwrite_sample_count": airwrite.get("sample_count"),
        "metrics": metrics,
        "warnings": warnings,
        "automatic_decision": None,
        "note": "Metrics support visual audit; they do not replace human orientation review.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare EMNIST and AirWrite model inputs")
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=preprocessing_alignment_settings.audit_output_dir,
    )
    arguments = parser.parse_args()
    emnist = read_report(arguments.audit_dir / "emnist_stats.json")
    airwrite = read_report(arguments.audit_dir / "airwrite_stats.json")
    comparison = build_comparison(emnist, airwrite)

    json_path = arguments.audit_dir / "comparison.json"
    csv_path = arguments.audit_dir / "comparison.csv"
    json_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("metric", "statistic", "emnist", "airwrite", "absolute_difference"),
        )
        writer.writeheader()
        for metric, values in comparison["metrics"].items():
            writer.writerow({"metric": metric, **values})

    print(f"Comparison: {json_path}")
    for warning in comparison["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
