import numpy as np

from app.inference.case_input_state import CaseInputState
from app.inference.case_label_resolver import CaseLabelResolver
from app.inference.case_mode_controller import CaseAction, CaseModeController
from app.inference.character_case_mode import CharacterCaseMode
from app.inference.identity_candidate import IdentityCandidate
from app.inference.prediction_policy import PredictionPolicy
from app.ml.letter_identity_labels import LETTER_IDENTITY_LABELS


def test_uncertain_candidates_keep_case_snapshot_when_global_mode_changes() -> None:
    state = CaseInputState(CharacterCaseMode.UPPERCASE)
    selection = state.create_selection()
    raw_candidates = (
        IdentityCandidate("d", 3, 0.42, 1),
        IdentityCandidate("o", 14, 0.39, 2),
        IdentityCandidate("g", 6, 0.10, 3),
    )
    status, margin = PredictionPolicy().evaluate(raw_candidates)
    case_resolver = CaseLabelResolver(
        LETTER_IDENTITY_LABELS,
        LETTER_IDENTITY_LABELS,
        tuple(label.upper() for label in LETTER_IDENTITY_LABELS),
    )
    rendered = tuple(
        case_resolver.resolve_candidate(candidate, selection.mode) for candidate in raw_candidates
    )

    CaseModeController(state).handle(CaseAction.SET_LOWERCASE)

    assert status.value == "UNCERTAIN"
    assert np.isclose(margin, 0.03)
    assert [candidate.rendered_character for candidate in rendered] == ["D", "O", "G"]
    assert selection.mode is CharacterCaseMode.UPPERCASE
