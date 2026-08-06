from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    identity_labels: tuple[str, ...]
    lowercase_display_labels: tuple[str, ...]
    uppercase_display_labels: tuple[str, ...]
    model_version: str
    task_type: str
    case_sensitive: bool
    case_source: str
    expected_input_shape: tuple[int, int, int]
    num_classes: int
    preprocessing_contract: dict[str, object]
    model_sha256: str | None = None
