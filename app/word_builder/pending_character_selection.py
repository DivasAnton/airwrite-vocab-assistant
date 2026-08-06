from dataclasses import dataclass

from app.inference.case_selection import CaseSelection
from app.inference.prediction_candidate import PredictionCandidate
from app.word_builder.exceptions import PendingSelectionError


@dataclass(frozen=True)
class PendingCharacterSelection:
    prediction_id: str
    candidates: tuple[PredictionCandidate, ...]
    case_selection: CaseSelection

    def __post_init__(self) -> None:
        if not self.prediction_id.strip():
            raise ValueError("prediction_id must not be empty")
        if not self.candidates:
            raise ValueError("pending selection must contain at least one candidate")
        expected_ranks = tuple(range(1, len(self.candidates) + 1))
        ranks = tuple(candidate.rank for candidate in self.candidates)
        if ranks != expected_ranks:
            raise ValueError("candidate ranks must be continuous and start at 1")
        rendered_characters = tuple(candidate.rendered_character for candidate in self.candidates)
        if len(set(rendered_characters)) != len(rendered_characters):
            raise ValueError("pending rendered characters must be unique")
        identities = tuple(candidate.identity for candidate in self.candidates)
        if len(set(identities)) != len(identities):
            raise ValueError("pending candidate identities must be unique")
        if any(
            candidate.case_mode is not self.case_selection.mode for candidate in self.candidates
        ):
            raise ValueError("pending candidates must match the frozen case selection")
        confidences = tuple(candidate.confidence for candidate in self.candidates)
        if any(first < second for first, second in zip(confidences, confidences[1:], strict=False)):
            raise ValueError("pending candidates must be ordered by descending confidence")

    def select_by_rank(self, rank: int) -> PredictionCandidate:
        if rank < 1 or rank > len(self.candidates):
            raise PendingSelectionError(
                f"Candidate rank must be between 1 and {len(self.candidates)}, got {rank}"
            )
        return self.candidates[rank - 1]
