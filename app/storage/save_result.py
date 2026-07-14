from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class SaveStatus(StrEnum):
    SAVED = "SAVED"
    SKIPPED_EMPTY = "SKIPPED_EMPTY"
    FAILED = "FAILED"
    DISABLED = "DISABLED"


@dataclass(frozen=True)
class SaveResult:
    status: SaveStatus
    file_path: Path | None
    saved_at: datetime | None
    message: str

    @property
    def success(self) -> bool:
        return self.status == SaveStatus.SAVED
