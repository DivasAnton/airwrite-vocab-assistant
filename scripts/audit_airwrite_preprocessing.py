import argparse
from pathlib import Path

import cv2

from app.preprocessing.aspect_ratio_resizer import AspectRatioResizer
from app.preprocessing.bounding_box_extractor import BoundingBoxExtractor
from app.preprocessing.exceptions import PreprocessingError
from app.preprocessing.foreground_normalizer import ForegroundNormalizer
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor
from app.preprocessing.model_input_contract import ModelInputContract
from app.preprocessing.preprocessing_auditor import PreprocessingAuditor, build_audit_grid
from app.preprocessing.preprocessing_source import PreprocessingSource
from app.utils.config import preprocessing_alignment_settings, settings


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
            output_width=settings.preprocess_output_width,
            output_height=settings.preprocess_output_height,
            content_width=settings.preprocess_content_width,
            content_height=settings.preprocess_content_height,
            center_of_mass=settings.preprocess_center_of_mass,
        ),
    )


def main() -> int:
    settings.validate_preprocessing_config()
    preprocessing_alignment_settings.validate()
    parser = argparse.ArgumentParser(description="Audit saved AirWrite canvas preprocessing")
    parser.add_argument("--input-dir", type=Path, default=settings.drawing_output_dir)
    parser.add_argument("--max-samples", type=int, default=260)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=preprocessing_alignment_settings.audit_output_dir,
    )
    parser.add_argument("--save-images", action="store_true")
    arguments = parser.parse_args()
    if arguments.max_samples <= 0:
        parser.error("--max-samples must be greater than 0")
    if not arguments.input_dir.exists():
        raise FileNotFoundError(f"AirWrite input directory does not exist: {arguments.input_dir}")

    contract = ModelInputContract(
        width=preprocessing_alignment_settings.input_width,
        height=preprocessing_alignment_settings.input_height,
        channels=preprocessing_alignment_settings.input_channels,
        background_value=preprocessing_alignment_settings.background_value,
        normalization_divisor=preprocessing_alignment_settings.normalization_divisor,
    )
    preprocessor = build_preprocessor()
    auditor = PreprocessingAuditor(contract, settings.preprocess_binary_threshold)
    records = []
    grid_samples = []
    failures: list[str] = []
    image_paths = sorted(arguments.input_dir.rglob("*.png"))[: arguments.max_samples]
    for image_path in image_paths:
        try:
            result = preprocessor.process_file(image_path)
        except PreprocessingError as error:
            failures.append(f"{image_path}: {error}")
            continue
        label = image_path.parent.name if len(image_path.parent.name) == 1 else None
        records.append(
            auditor.analyze(
                result.processed_image,
                source=PreprocessingSource.AIRWRITE_CANVAS,
                label=label,
            )
        )
        grid_samples.append((label or image_path.stem[:8], result.processed_image))

    if not records:
        raise RuntimeError("No valid AirWrite PNG images were available for audit")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    auditor.write_reports(
        records,
        json_path=arguments.output_dir / "airwrite_stats.json",
        csv_path=arguments.output_dir / "airwrite_stats.csv",
    )
    if arguments.save_images or preprocessing_alignment_settings.save_audit_images:
        grid_path = arguments.output_dir / "airwrite_processed_grid.png"
        if not cv2.imwrite(str(grid_path), build_audit_grid(grid_samples)):
            raise RuntimeError(f"Could not write AirWrite audit grid: {grid_path}")

    print(f"AirWrite files found: {len(image_paths)}")
    print(f"AirWrite samples audited: {len(records)}")
    print(f"Rejected samples: {len(failures)}")
    print(f"Report: {arguments.output_dir / 'airwrite_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
