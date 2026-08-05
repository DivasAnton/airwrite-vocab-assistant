import csv
from dataclasses import dataclass, replace
from pathlib import Path

MANIFEST_COLUMNS = (
    "image_path",
    "label",
    "label_index",
    "source",
    "writer_id",
    "session_id",
    "split",
)


@dataclass(frozen=True)
class DatasetManifestEntry:
    image_path: Path
    label: str
    label_index: int
    source: str
    writer_id: str | None = None
    session_id: str | None = None
    split: str | None = None

    def with_split(self, split: str) -> "DatasetManifestEntry":
        return replace(self, split=split)

    def to_csv_row(self) -> dict[str, str | int]:
        return {
            "image_path": self.image_path.as_posix(),
            "label": self.label,
            "label_index": self.label_index,
            "source": self.source,
            "writer_id": self.writer_id or "",
            "session_id": self.session_id or "",
            "split": self.split or "",
        }


def write_manifest(entries: list[DatasetManifestEntry], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(entry.to_csv_row() for entry in entries)


def read_manifest(manifest_path: Path) -> list[DatasetManifestEntry]:
    entries: list[DatasetManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != list(MANIFEST_COLUMNS):
            raise ValueError(
                f"Manifest columns must be {list(MANIFEST_COLUMNS)}, got {reader.fieldnames}"
            )
        for row in reader:
            entries.append(
                DatasetManifestEntry(
                    image_path=Path(row["image_path"]),
                    label=row["label"],
                    label_index=int(row["label_index"]),
                    source=row["source"],
                    writer_id=row["writer_id"] or None,
                    session_id=row["session_id"] or None,
                    split=row["split"] or None,
                )
            )
    return entries
