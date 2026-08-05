from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    labels: tuple[str, ...]
    model_version: str
    expected_input_shape: tuple[int, int, int]
    num_classes: int
    preprocessing_contract: dict[str, object]
    model_sha256: str | None = None
