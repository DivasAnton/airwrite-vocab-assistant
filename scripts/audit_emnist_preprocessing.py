import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from app.preprocessing.emnist_source_adapter import EMNISTSourceAdapter
from app.preprocessing.model_input_contract import ModelInputContract
from app.preprocessing.model_input_preprocessor import ModelInputPreprocessor
from app.preprocessing.preprocessing_auditor import PreprocessingAuditor, build_audit_grid
from app.preprocessing.preprocessing_source import PreprocessingSource
from app.utils.config import preprocessing_alignment_settings


def identity_label(value: int) -> str:
    if 1 <= value <= 26:
        return chr(ord("A") + value - 1)
    raise ValueError(f"Unexpected EMNIST Letters label: {value}")


def load_samples(
    dataset_name: str,
    data_dir: Path,
    samples_per_class: int,
) -> list[tuple[str, NDArray[np.uint8]]]:
    try:
        import tensorflow_datasets as tfds
    except ImportError as error:
        raise RuntimeError(
            "tensorflow-datasets is required; install requirements-ml.txt"
        ) from error

    dataset = tfds.load(
        dataset_name,
        split="train",
        data_dir=str(data_dir),
        as_supervised=True,
        shuffle_files=False,
    )
    selected: dict[str, list[NDArray[np.uint8]]] = defaultdict(list)
    for image, label in tfds.as_numpy(dataset):
        identity = identity_label(int(label))
        if len(selected[identity]) < samples_per_class:
            selected[identity].append(np.asarray(image, dtype=np.uint8))
        if len(selected) == 26 and all(
            len(images) >= samples_per_class for images in selected.values()
        ):
            break

    missing = [
        label for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if len(selected[label]) < samples_per_class
    ]
    if missing:
        raise RuntimeError(f"Not enough EMNIST samples for identities: {', '.join(missing)}")
    return [
        (label, image)
        for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for image in selected[label][:samples_per_class]
    ]


def write_grid(path: Path, samples: list[tuple[str, NDArray[np.uint8]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), build_audit_grid(samples)):
        raise RuntimeError(f"Could not write audit grid: {path}")


def enrich_report(path: Path, metadata: dict[str, Any]) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    report.update(metadata)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    preprocessing_alignment_settings.validate()
    parser = argparse.ArgumentParser(description="Audit TFDS EMNIST Letters preprocessing")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=preprocessing_alignment_settings.audit_samples_per_class,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=preprocessing_alignment_settings.audit_output_dir,
    )
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument(
        "--visual-audit-status",
        choices=("pending_manual_review", "verified_transpose", "verified_none"),
        default="pending_manual_review",
        help="Set only after a human compares the generated orientation grids.",
    )
    parser.add_argument(
        "--transpose",
        choices=("configured", "true", "false"),
        default="configured",
    )
    arguments = parser.parse_args()
    if arguments.samples_per_class <= 0:
        parser.error("--samples-per-class must be greater than 0")

    transpose_images = preprocessing_alignment_settings.emnist_transpose_images
    if arguments.transpose != "configured":
        transpose_images = arguments.transpose == "true"
    contract = ModelInputContract(
        width=preprocessing_alignment_settings.input_width,
        height=preprocessing_alignment_settings.input_height,
        channels=preprocessing_alignment_settings.input_channels,
        background_value=preprocessing_alignment_settings.background_value,
        normalization_divisor=preprocessing_alignment_settings.normalization_divisor,
    )
    raw_samples = load_samples(
        preprocessing_alignment_settings.emnist_dataset_name,
        preprocessing_alignment_settings.emnist_data_dir,
        arguments.samples_per_class,
    )
    no_transpose_adapter = EMNISTSourceAdapter(contract, transpose_images=False)
    transpose_adapter = EMNISTSourceAdapter(contract, transpose_images=True)
    selected_adapter = EMNISTSourceAdapter(
        contract,
        transpose_images=transpose_images,
        preserve_grayscale=preprocessing_alignment_settings.emnist_preserve_grayscale,
    )
    common_preprocessor = ModelInputPreprocessor(contract)
    auditor = PreprocessingAuditor(contract)

    raw_grid_samples: list[tuple[str, NDArray[np.uint8]]] = []
    no_transpose_samples: list[tuple[str, NDArray[np.uint8]]] = []
    transposed_samples: list[tuple[str, NDArray[np.uint8]]] = []
    processed_samples: list[tuple[str, NDArray[np.uint8]]] = []
    records = []
    for label, raw_image in raw_samples:
        raw_2d = no_transpose_adapter.adapt(raw_image)
        no_transpose = no_transpose_adapter.adapt(raw_image)
        transposed = transpose_adapter.adapt(raw_image)
        prepared = selected_adapter.adapt(raw_image)
        result = common_preprocessor.process(
            prepared,
            source=PreprocessingSource.EMNIST_LETTERS,
            orientation_transform=selected_adapter.orientation_transform,
            original_shape=tuple(raw_image.shape),
        )
        raw_grid_samples.append((label, raw_2d))
        no_transpose_samples.append((label, no_transpose))
        transposed_samples.append((label, transposed))
        processed_samples.append((label, result.processed_image))
        records.append(
            auditor.analyze(
                result.processed_image,
                source=PreprocessingSource.EMNIST_LETTERS,
                label=label,
            )
        )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = arguments.output_dir / "emnist_stats.json"
    auditor.write_reports(
        records,
        json_path=report_path,
        csv_path=arguments.output_dir / "emnist_stats.csv",
    )
    enrich_report(
        report_path,
        {
            "dataset_name": preprocessing_alignment_settings.emnist_dataset_name,
            "orientation_transform": selected_adapter.orientation_transform,
            "intensity_inversion": False,
            "visual_audit_status": arguments.visual_audit_status,
        },
    )
    if arguments.save_images or preprocessing_alignment_settings.save_audit_images:
        write_grid(arguments.output_dir / "emnist_raw_grid.png", raw_grid_samples)
        write_grid(arguments.output_dir / "emnist_no_transpose_grid.png", no_transpose_samples)
        write_grid(arguments.output_dir / "emnist_transposed_grid.png", transposed_samples)
        write_grid(arguments.output_dir / "emnist_processed_grid.png", processed_samples)

    print(f"EMNIST samples audited: {len(records)}")
    print(f"Orientation transform: {selected_adapter.orientation_transform}")
    print(f"Report: {report_path}")
    print(f"Visual audit status: {arguments.visual_audit_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
