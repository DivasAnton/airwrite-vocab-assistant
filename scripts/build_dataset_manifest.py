import csv
import json
from collections import Counter
from pathlib import Path

from app.ml.dataset_manifest import DatasetManifestEntry, write_manifest
from app.ml.dataset_scanner import DatasetScanIssue, DatasetScanner
from app.ml.dataset_splitter import DatasetSplitter
from app.ml.dataset_validator import (
    DatasetValidationIssue,
    DatasetValidationReport,
    DatasetValidator,
)
from app.preprocessing.aspect_ratio_resizer import AspectRatioResizer
from app.preprocessing.bounding_box_extractor import BoundingBoxExtractor
from app.preprocessing.foreground_normalizer import ForegroundNormalizer
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor
from app.utils.config import PROJECT_ROOT, settings, training_settings


def build_preprocessor() -> HandwritingPreprocessor:
    return HandwritingPreprocessor(
        normalizer=ForegroundNormalizer(
            binary_threshold=settings.preprocess_binary_threshold,
            invert_input=settings.preprocess_invert_input,
        ),
        extractor=BoundingBoxExtractor(
            crop_padding=settings.preprocess_crop_padding,
            min_foreground_pixels=settings.preprocess_min_foreground_pixels,
        ),
        resizer=AspectRatioResizer(
            output_width=training_settings.input_width,
            output_height=training_settings.input_height,
            content_width=settings.preprocess_content_width,
            content_height=settings.preprocess_content_height,
            center_of_mass=settings.preprocess_center_of_mass,
        ),
    )


def write_validation_errors(
    scan_issues: list[DatasetScanIssue],
    validation_issues: list[DatasetValidationIssue],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=("image_path", "error_type", "message"))
        writer.writeheader()
        for issue in scan_issues:
            writer.writerow(
                {
                    "image_path": issue.image_path.as_posix(),
                    "error_type": issue.error_type,
                    "message": issue.message,
                }
            )
        for validation_issue in validation_issues:
            writer.writerow(
                {
                    "image_path": validation_issue.image_path.as_posix(),
                    "error_type": validation_issue.error_type,
                    "message": validation_issue.message,
                }
            )


def build_report(
    scanned_count: int,
    validation: DatasetValidationReport,
    split_entries: list[DatasetManifestEntry],
    scan_issues: list[DatasetScanIssue],
) -> dict[str, object]:
    class_counts = Counter(entry.label for entry in validation.valid_entries)
    split_counts = Counter(entry.split for entry in split_entries)
    source_counts = Counter(entry.source for entry in validation.valid_entries)
    split_class_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "validation", "test"):
        split_class_counts[split_name] = dict(
            sorted(
                Counter(entry.label for entry in split_entries if entry.split == split_name).items()
            )
        )

    smallest_count = min(class_counts.values(), default=0)
    largest_count = max(class_counts.values(), default=0)
    imbalance_ratio = largest_count / smallest_count if smallest_count else None
    issue_counts = Counter(issue.error_type for issue in scan_issues)
    issue_counts.update(issue.error_type for issue in validation.issues)
    return {
        "dataset_root": training_settings.dataset_root.relative_to(PROJECT_ROOT).as_posix(),
        "scanned_images": scanned_count,
        "valid_unique_images": len(validation.valid_entries),
        "invalid_or_duplicate_images": len(validation.issues),
        "structure_issues": len(scan_issues),
        "exact_duplicates": validation.duplicate_count,
        "class_counts": dict(sorted(class_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "smallest_class_count": smallest_count,
        "largest_class_count": largest_count,
        "imbalance_ratio": imbalance_ratio,
        "split_counts": {
            split_name: split_counts[split_name] for split_name in ("train", "validation", "test")
        },
        "split_class_counts": split_class_counts,
        "issue_counts": dict(sorted(issue_counts.items())),
        "random_seed": training_settings.random_seed,
        "ratios": {
            "train": training_settings.train_ratio,
            "validation": training_settings.validation_ratio,
            "test": training_settings.test_ratio,
        },
    }


def main() -> int:
    training_settings.validate()
    scanner = DatasetScanner(
        training_settings.dataset_root,
        source="airwrite",
        project_root=PROJECT_ROOT,
    )
    entries = scanner.scan()
    validator = DatasetValidator(
        preprocessor=build_preprocessor(),
        project_root=PROJECT_ROOT,
        expected_shape=(training_settings.input_height, training_settings.input_width),
        require_grayscale=training_settings.input_channels == 1,
        required_extension=".png",
    )
    validation = validator.validate(entries)

    splitter = DatasetSplitter(
        train_ratio=training_settings.train_ratio,
        validation_ratio=training_settings.validation_ratio,
        test_ratio=training_settings.test_ratio,
        random_seed=training_settings.random_seed,
    )
    split_entries = splitter.split(validation.valid_entries)

    write_manifest(validation.valid_entries, training_settings.manifest_path)
    write_manifest(split_entries, training_settings.split_path)
    manifest_dir = training_settings.manifest_path.parent
    errors_path = manifest_dir / "dataset_validation_errors.csv"
    report_path = manifest_dir / "dataset_report.json"
    write_validation_errors(scanner.issues, validation.issues, errors_path)
    report = build_report(len(entries), validation, split_entries, scanner.issues)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Scanned images: {len(entries)}")
    print(f"Valid unique images: {len(validation.valid_entries)}")
    print(f"Validation issues: {len(validation.issues)}")
    print(f"Structure issues: {len(scanner.issues)}")
    split_counts = Counter(entry.split for entry in split_entries)
    print(
        "Splits: "
        f"train={split_counts['train']}, "
        f"validation={split_counts['validation']}, "
        f"test={split_counts['test']}"
    )
    print(f"Manifest: {training_settings.manifest_path}")
    print(f"Split manifest: {training_settings.split_path}")
    print(f"Validation report: {errors_path}")
    print(f"Dataset report: {report_path}")
    blocking_issues = [
        issue for issue in validation.issues if issue.error_type != "exact_duplicate"
    ]
    return 1 if scanner.issues or blocking_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
