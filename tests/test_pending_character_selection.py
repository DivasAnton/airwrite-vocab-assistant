import pytest

from app.inference.case_selection import CaseSelection
from app.inference.character_case_mode import CharacterCaseMode
from app.inference.prediction_candidate import PredictionCandidate
from app.word_builder.exceptions import PendingSelectionError
from app.word_builder.pending_character_selection import PendingCharacterSelection


def candidates() -> tuple[PredictionCandidate, ...]:
    return (
        PredictionCandidate("O", 14, 0.45, 1),
        PredictionCandidate("D", 3, 0.41, 2),
        PredictionCandidate("G", 6, 0.08, 3),
    )


UPPERCASE_SELECTION = CaseSelection(CharacterCaseMode.UPPERCASE, False)


def test_selects_each_valid_rank() -> None:
    selection = PendingCharacterSelection("pred_1", candidates(), UPPERCASE_SELECTION)

    assert selection.select_by_rank(1).label == "O"
    assert selection.select_by_rank(2).label == "D"
    assert selection.select_by_rank(3).label == "G"


@pytest.mark.parametrize("rank", [0, 4])
def test_rejects_rank_outside_candidates(rank: int) -> None:
    selection = PendingCharacterSelection("pred_1", candidates(), UPPERCASE_SELECTION)

    with pytest.raises(PendingSelectionError):
        selection.select_by_rank(rank)


def test_rejects_duplicate_labels() -> None:
    duplicate = (
        PredictionCandidate("D", 3, 0.60, 1),
        PredictionCandidate("D", 3, 0.40, 2),
    )

    with pytest.raises(ValueError, match="unique"):
        PendingCharacterSelection("pred_1", duplicate, UPPERCASE_SELECTION)


def test_rejects_increasing_confidence_order() -> None:
    unordered = (
        PredictionCandidate("D", 3, 0.40, 1),
        PredictionCandidate("O", 14, 0.45, 2),
    )

    with pytest.raises(ValueError, match="descending"):
        PendingCharacterSelection("pred_1", unordered, UPPERCASE_SELECTION)


def test_rejects_candidates_that_do_not_match_frozen_case_selection() -> None:
    mixed_case = (
        PredictionCandidate("D", 3, 0.60, 1),
        PredictionCandidate("o", 14, 0.40, 2),
    )

    with pytest.raises(ValueError, match="frozen case"):
        PendingCharacterSelection("pred_1", mixed_case, UPPERCASE_SELECTION)
