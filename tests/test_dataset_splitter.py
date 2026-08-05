from pathlib import Path

from app.ml.dataset_manifest import DatasetManifestEntry
from app.ml.dataset_splitter import DatasetSplitter
from app.ml.labels import CHARACTER_LABELS


def make_entries(samples_per_label: int = 20) -> list[DatasetManifestEntry]:
    return [
        DatasetManifestEntry(
            image_path=Path("data/raw_airwrite") / label / f"{sample_index:03d}.png",
            label=label,
            label_index=label_index,
            source="airwrite",
        )
        for label_index, label in enumerate(CHARACTER_LABELS)
        for sample_index in range(samples_per_label)
    ]


def test_splitter_is_reproducible_and_keeps_every_label_in_every_split() -> None:
    splitter = DatasetSplitter(random_seed=42)

    first = splitter.split(make_entries())
    second = splitter.split(make_entries())

    assert first == second
    assert len({entry.image_path for entry in first}) == len(first)
    for split_name in ("train", "validation", "test"):
        assert {entry.label for entry in first if entry.split == split_name} == set(
            CHARACTER_LABELS
        )


def test_splitter_uses_expected_per_class_counts() -> None:
    entries = DatasetSplitter().split(make_entries(samples_per_label=20))

    assert sum(entry.split == "train" for entry in entries) == 14 * 26
    assert sum(entry.split == "validation" for entry in entries) == 3 * 26
    assert sum(entry.split == "test" for entry in entries) == 3 * 26
