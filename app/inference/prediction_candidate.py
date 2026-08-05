from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PredictionCandidate:
    label: str
    class_index: int
    confidence: float
    rank: int

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label must not be empty")
        if self.class_index < 0:
            raise ValueError("class_index must be greater than or equal to 0")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0.0 and 1.0")
        if self.rank < 1:
            raise ValueError("rank must be greater than or equal to 1")
