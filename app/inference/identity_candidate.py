from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class IdentityCandidate:
    identity: str
    class_index: int
    confidence: float
    rank: int

    def __post_init__(self) -> None:
        if len(self.identity) != 1 or self.identity not in "abcdefghijklmnopqrstuvwxyz":
            raise ValueError("identity must be one canonical lowercase letter from a to z")
        if not 0 <= self.class_index < 26:
            raise ValueError("class_index must be between 0 and 25")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0.0 and 1.0")
        if self.rank < 1:
            raise ValueError("rank must be greater than or equal to 1")
