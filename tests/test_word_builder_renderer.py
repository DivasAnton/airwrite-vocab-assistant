import numpy as np

from app.inference.case_input_state import CaseInputState
from app.inference.case_selection import CaseSelection
from app.inference.character_case_mode import CharacterCaseMode
from app.inference.prediction_candidate import PredictionCandidate
from app.word_builder.pending_character_selection import PendingCharacterSelection
from app.word_builder.word_action import WordAction
from app.word_builder.word_builder_renderer import WordBuilderRenderer
from app.word_builder.word_builder_result import WordBuilderResult
from app.word_builder.word_builder_state import WordBuilderState


def result_with_pending() -> WordBuilderResult:
    pending = PendingCharacterSelection(
        "pred_d",
        (
            PredictionCandidate("O", 14, 0.45, 1),
            PredictionCandidate("D", 3, 0.41, 2),
            PredictionCandidate("G", 6, 0.08, 3),
        ),
        CaseSelection(CharacterCaseMode.UPPERCASE, False),
    )
    return WordBuilderResult(
        action=WordAction.PENDING_SELECTION_CREATED,
        state=WordBuilderState.AWAITING_SELECTION,
        current_word="WOR",
        canonical_word="wor",
        committed_entry=None,
        pending_selection=pending,
        confirmed_word=None,
        message="Choose a pending character or cancel it",
    )


def test_renderer_formats_empty_word() -> None:
    result = WordBuilderResult(
        action=WordAction.NONE,
        state=WordBuilderState.EMPTY,
        current_word="",
        canonical_word="",
        committed_entry=None,
        pending_selection=None,
        confirmed_word=None,
        message="Ready",
    )

    lines = WordBuilderRenderer(30).format_lines(result)

    assert lines[0] == "Word: _"
    assert lines[1] == "Length: 0/30"


def test_renderer_formats_pending_candidates() -> None:
    case_state = CaseInputState(CharacterCaseMode.LOWERCASE)
    lines = WordBuilderRenderer(30).format_lines(
        result_with_pending(),
        case_state=case_state,
    )

    assert "Word: WOR" in lines
    assert "1. O 45.0%" in lines
    assert "2. D 41.0%" in lines
    assert "3. G 8.0%" in lines
    assert "Mode: lowercase" in lines
    assert "Pending mode: UPPERCASE" in lines


def test_renderer_preserves_word_case_and_describes_effective_shift_mode() -> None:
    case_state = CaseInputState(CharacterCaseMode.LOWERCASE, shift_next=True)
    result = WordBuilderResult(
        action=WordAction.NONE,
        state=WordBuilderState.BUILDING,
        current_word="AirWrite",
        canonical_word="airwrite",
        committed_entry=None,
        pending_selection=None,
        confirmed_word=None,
        message="Ready",
    )

    lines = WordBuilderRenderer(30).format_lines(result, case_state=case_state)

    assert "Word: AirWrite" in lines
    assert "Mode: lowercase" in lines
    assert "Next character: UPPERCASE" in lines


def test_renderer_keeps_shape_and_does_not_mutate_frame() -> None:
    renderer = WordBuilderRenderer(30)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    original = frame.copy()

    output = renderer.render(frame, result_with_pending(), 1000, 1100)

    assert output.shape == frame.shape
    assert np.array_equal(frame, original)
    assert not np.array_equal(output, frame)


def test_disabled_renderer_returns_frame_copy() -> None:
    renderer = WordBuilderRenderer(30, enabled=False)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    output = renderer.render(frame, result_with_pending(), 1000, 1100)

    assert output is not frame
    assert np.array_equal(output, frame)
