import json

from app.ml.emnist_dataset_auditor import EMNISTDatasetAuditor
from app.ml.emnist_letters_dataset import EMNISTLettersDataset
from app.ml.emnist_split_builder import EMNISTSplitBuilder, EMNISTSplitIndices
from app.utils.config import emnist_training_settings


def load_official_dataset() -> EMNISTLettersDataset:
    settings = emnist_training_settings
    settings.validate()
    return EMNISTLettersDataset.load(
        train_images_path=settings.train_images_path,
        train_labels_path=settings.train_labels_path,
        test_images_path=settings.test_images_path,
        test_labels_path=settings.test_labels_path,
    )


def ensure_split(dataset: EMNISTLettersDataset) -> EMNISTSplitIndices:
    settings = emnist_training_settings
    path = settings.split_indices_path
    if path.is_file():
        indices = EMNISTSplitIndices.load(path)
        if indices.random_seed != settings.random_seed:
            raise ValueError("Saved split seed differs from EMNIST_RANDOM_SEED")
        if abs(indices.validation_ratio - settings.validation_ratio) > 1e-12:
            raise ValueError("Saved split ratio differs from EMNIST_VALIDATION_RATIO")
    else:
        indices = EMNISTSplitBuilder(
            validation_ratio=settings.validation_ratio,
            random_seed=settings.random_seed,
        ).build(dataset.official_train.labels)
        indices.save(path)
    EMNISTSplitBuilder.validate(indices, len(dataset.official_train.labels))
    return indices


def main() -> None:
    settings = emnist_training_settings
    dataset = load_official_dataset()
    paths = {
        "train_images": settings.train_images_path,
        "train_labels": settings.train_labels_path,
        "test_images": settings.test_images_path,
        "test_labels": settings.test_labels_path,
    }
    report = EMNISTDatasetAuditor().audit(dataset, paths).to_payload()
    indices = ensure_split(dataset)
    report["validation_split"] = {
        "random_seed": indices.random_seed,
        "validation_ratio": indices.validation_ratio,
        "train_sample_count": len(indices.train_indices),
        "validation_sample_count": len(indices.validation_indices),
        "official_test_sample_count": len(dataset.official_test.labels),
        "indices_path": settings.split_indices_path.as_posix(),
    }
    output_path = settings.data_root.parent / "audit.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Audit passed: {output_path}")
    print(
        "Samples: "
        f"train={len(indices.train_indices):,}, "
        f"validation={len(indices.validation_indices):,}, "
        f"official_test={len(dataset.official_test.labels):,}"
    )
    print("Labels: raw 1-26 -> mapped 0-25 (26 identities)")


if __name__ == "__main__":
    main()
