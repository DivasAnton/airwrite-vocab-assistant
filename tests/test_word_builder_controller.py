from app.inference.prediction_candidate import PredictionCandidate
from app.inference.prediction_result import PredictionResult
from app.inference.prediction_status import PredictionStatus
from app.word_builder.character_source import CharacterSource
from app.word_builder.word_action import WordAction
from app.word_builder.word_builder import WordBuilder
from app.word_builder.word_builder_controller import WordBuilderController
from app.word_builder.word_builder_state import WordBuilderState


def prediction(
    prediction_id: str,
    status: PredictionStatus,
    labels: tuple[str, ...] = ("C", "G", "O"),
    confidences: tuple[float, ...] = (0.80, 0.12, 0.08),
) -> PredictionResult:
    candidates = tuple(
        PredictionCandidate(label, ord(label) - ord("A"), confidence, rank)
        for rank, (label, confidence) in enumerate(zip(labels, confidences, strict=True), start=1)
    )
    has_prediction = status in {PredictionStatus.ACCEPTED, PredictionStatus.UNCERTAIN}
    return PredictionResult(
        status=status,
        top_prediction=candidates[0] if has_prediction else None,
        candidates=candidates if has_prediction else (),
        confidence_margin=confidences[0] - confidences[1] if has_prediction else None,
        inference_time_ms=5.0 if has_prediction else None,
        model_version="0.1.0",
        message=status.value,
        prediction_id=prediction_id,
    )


def test_accepted_predictions_build_cat() -> None:
    controller = WordBuilderController(WordBuilder())

    for identifier, character in (("pred_c", "C"), ("pred_a", "A"), ("pred_t", "T")):
        result = controller.handle_prediction(
            prediction(identifier, PredictionStatus.ACCEPTED, (character, "Z", "Y"))
        )

    assert result.current_word == "CAT"
    assert controller.word_builder.entries[-1].source == CharacterSource.AUTO_ACCEPTED


def test_uncertain_prediction_waits_then_appends_selected_rank() -> None:
    controller = WordBuilderController(WordBuilder())
    for identifier, character in (("pred_w", "W"), ("pred_o", "O"), ("pred_r", "R")):
        controller.handle_prediction(
            prediction(identifier, PredictionStatus.ACCEPTED, (character, "Z", "Y"))
        )

    pending = controller.handle_prediction(
        prediction(
            "pred_d",
            PredictionStatus.UNCERTAIN,
            ("O", "D", "G"),
            (0.44, 0.40, 0.16),
        )
    )
    selected = controller.select_candidate(2)

    assert pending.current_word == "WOR"
    assert pending.state == WordBuilderState.AWAITING_SELECTION
    assert selected.current_word == "WORD"
    assert selected.action == WordAction.PENDING_SELECTION_RESOLVED
    assert controller.word_builder.entries[-1].source == CharacterSource.USER_SELECTED
    assert selected.pending_selection is None


def test_duplicate_event_is_ignored_but_two_p_events_build_pp() -> None:
    controller = WordBuilderController(WordBuilder())
    first = prediction("pred_1", PredictionStatus.ACCEPTED, ("P", "F", "R"))
    second = prediction("pred_2", PredictionStatus.ACCEPTED, ("P", "F", "R"))

    controller.handle_prediction(first)
    duplicate = controller.handle_prediction(first)
    repeated = controller.handle_prediction(second)

    assert duplicate.action == WordAction.DUPLICATE_IGNORED
    assert repeated.current_word == "PP"


def test_cancel_pending_marks_event_processed_and_keeps_word() -> None:
    controller = WordBuilderController(WordBuilder())
    uncertain = prediction("pred_x", PredictionStatus.UNCERTAIN)
    controller.handle_prediction(uncertain)

    cancelled = controller.cancel_pending()
    duplicate = controller.handle_prediction(uncertain)

    assert cancelled.current_word == ""
    assert cancelled.action == WordAction.PENDING_SELECTION_CANCELLED
    assert duplicate.action == WordAction.DUPLICATE_IGNORED


def test_empty_failed_and_missing_id_do_not_modify_word() -> None:
    controller = WordBuilderController(WordBuilder())

    skipped = controller.handle_prediction(prediction("skip", PredictionStatus.SKIPPED_EMPTY))
    failed = controller.handle_prediction(prediction("fail", PredictionStatus.FAILED))
    missing_id = prediction("id", PredictionStatus.ACCEPTED)
    missing_id = PredictionResult(
        status=missing_id.status,
        top_prediction=missing_id.top_prediction,
        candidates=missing_id.candidates,
        confidence_margin=missing_id.confidence_margin,
        inference_time_ms=missing_id.inference_time_ms,
        model_version=missing_id.model_version,
        message=missing_id.message,
    )
    invalid = controller.handle_prediction(missing_id)

    assert skipped.action == WordAction.NONE
    assert failed.action == WordAction.ERROR
    assert invalid.action == WordAction.ERROR
    assert invalid.current_word == ""


def test_auto_append_disabled_creates_pending_for_accepted_prediction() -> None:
    controller = WordBuilderController(WordBuilder(), auto_append_accepted=False)

    result = controller.handle_prediction(prediction("pred_c", PredictionStatus.ACCEPTED))

    assert result.action == WordAction.PENDING_SELECTION_CREATED
    assert result.current_word == ""


def test_cat_integration_confirms_three_entries() -> None:
    controller = WordBuilderController(WordBuilder())
    for identifier, character in (("1", "C"), ("2", "A"), ("3", "T")):
        controller.handle_prediction(
            prediction(identifier, PredictionStatus.ACCEPTED, (character, "Z", "Y"))
        )

    result = controller.confirm(999)

    assert result.confirmed_word is not None
    assert result.confirmed_word.word == "CAT"
    assert len(result.confirmed_word.entries) == 3


def test_confirm_is_blocked_while_selection_is_pending() -> None:
    controller = WordBuilderController(WordBuilder())
    controller.handle_prediction(prediction("pred_c", PredictionStatus.ACCEPTED))
    controller.handle_prediction(prediction("pred_x", PredictionStatus.UNCERTAIN))

    result = controller.confirm(1)

    assert result.action == WordAction.ERROR
    assert result.state == WordBuilderState.AWAITING_SELECTION
