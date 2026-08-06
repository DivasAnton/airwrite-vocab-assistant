from app.inference.case_selection import CaseSelection
from app.inference.character_case_mode import CharacterCaseMode
from app.inference.prediction_candidate import PredictionCandidate
from app.inference.prediction_result import PredictionResult
from app.inference.prediction_status import PredictionStatus
from app.word_builder.word_action import WordAction
from app.word_builder.word_builder import WordBuilder
from app.word_builder.word_builder_controller import WordBuilderController


def candidate(
    identity: str,
    confidence: float,
    rank: int,
    mode: CharacterCaseMode,
) -> PredictionCandidate:
    rendered = identity if mode is CharacterCaseMode.LOWERCASE else identity.upper()
    return PredictionCandidate(
        label=rendered,
        class_index=ord(identity) - ord("a"),
        confidence=confidence,
        rank=rank,
        identity=identity,
        rendered_character=rendered,
        case_mode=mode,
    )


def prediction(
    prediction_id: str,
    identities: tuple[str, ...],
    mode: CharacterCaseMode,
    *,
    status: PredictionStatus = PredictionStatus.ACCEPTED,
    shift: bool = False,
) -> PredictionResult:
    confidences = (0.8, 0.15, 0.05)[: len(identities)]
    candidates = tuple(
        candidate(identity, confidence, rank, mode)
        for rank, (identity, confidence) in enumerate(
            zip(identities, confidences, strict=True), start=1
        )
    )
    return PredictionResult(
        status=status,
        top_prediction=candidates[0],
        candidates=candidates,
        confidence_margin=(
            confidences[0] - confidences[1] if len(confidences) > 1 else confidences[0]
        ),
        inference_time_ms=1.0,
        model_version="1.0.0",
        message=status.value,
        prediction_id=prediction_id,
        case_selection=CaseSelection(mode, shift),
    )


def test_controller_builds_cat_and_airwrite_without_normalizing_case() -> None:
    for word in ("Cat", "AirWrite"):
        controller = WordBuilderController(WordBuilder())
        for index, character in enumerate(word):
            mode = (
                CharacterCaseMode.UPPERCASE
                if character in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                else CharacterCaseMode.LOWERCASE
            )
            controller.handle_prediction(prediction(f"pred_{index}", (character.casefold(),), mode))

        confirmed = controller.confirm(10)
        assert confirmed.confirmed_word is not None
        assert confirmed.confirmed_word.word == word
        assert confirmed.confirmed_word.canonical_word == word.casefold()


def test_uncertain_selection_keeps_frozen_case_when_global_mode_changes() -> None:
    controller = WordBuilderController(WordBuilder())
    pending = controller.handle_prediction(
        prediction(
            "pred_d",
            ("o", "d", "g"),
            CharacterCaseMode.UPPERCASE,
            status=PredictionStatus.UNCERTAIN,
        )
    )

    selected = controller.select_candidate(2)

    assert pending.action is WordAction.PENDING_SELECTION_CREATED
    assert pending.pending_selection is not None
    assert [item.rendered_character for item in pending.pending_selection.candidates] == [
        "O",
        "D",
        "G",
    ]
    assert selected.current_word == "D"
    assert selected.committed_entry is not None
    assert selected.committed_entry.identity == "d"


def test_failed_and_empty_predictions_do_not_commit_entries() -> None:
    controller = WordBuilderController(WordBuilder())
    for status in (PredictionStatus.FAILED, PredictionStatus.SKIPPED_EMPTY):
        result = PredictionResult(
            status=status,
            top_prediction=None,
            candidates=(),
            confidence_margin=None,
            inference_time_ms=None,
            model_version="1.0.0",
            message=status.value,
            prediction_id=f"pred_{status.value}",
            case_selection=CaseSelection(CharacterCaseMode.UPPERCASE, True),
        )
        handled = controller.handle_prediction(result)
        assert handled.current_word == ""
        assert handled.committed_entry is None
