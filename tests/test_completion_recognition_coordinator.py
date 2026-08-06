from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.drawing.air_canvas import AirCanvas
from app.drawing.drawing_state import DrawingState
from app.inference.case_input_state import CaseInputState
from app.inference.case_selection import CaseSelection
from app.inference.character_case_mode import CharacterCaseMode
from app.inference.completion_recognition_coordinator import CompletionRecognitionCoordinator
from app.inference.prediction_candidate import PredictionCandidate
from app.inference.prediction_result import PredictionResult
from app.inference.prediction_status import PredictionStatus
from app.storage.save_result import SaveResult, SaveStatus


class FakeSaveCoordinator:
    def __init__(self) -> None:
        self.calls = 0
        self.snapshots: list[NDArray[np.uint8]] = []

    def save_snapshot(
        self,
        snapshot: NDArray[np.uint8],
        *,
        is_empty: bool,
        manual: bool,
    ) -> SaveResult:
        assert not manual
        self.calls += 1
        self.snapshots.append(snapshot)
        return SaveResult(
            SaveStatus.SKIPPED_EMPTY if is_empty else SaveStatus.SAVED,
            None if is_empty else Path("drawing.png"),
            datetime.now(),
            "saved",
        )


class FakeRecognitionService:
    def __init__(self) -> None:
        self.calls = 0
        self.snapshots: list[NDArray[np.uint8]] = []

    def recognize(
        self,
        snapshot: NDArray[np.uint8],
        *,
        case_selection: CaseSelection,
        prediction_id: str,
    ) -> PredictionResult:
        self.calls += 1
        self.snapshots.append(snapshot)
        candidate = PredictionCandidate("A", 0, 0.8, 1)
        return PredictionResult(
            PredictionStatus.ACCEPTED,
            candidate,
            (candidate,),
            0.8,
            1.0,
            "0.1.0",
            "accepted",
            prediction_id,
            case_selection,
            False,
        )


class CountingCanvas(AirCanvas):
    def __init__(self) -> None:
        super().__init__(80, 60)
        self.snapshot_calls = 0

    def get_image(self, copy: bool = True) -> NDArray[np.uint8]:
        self.snapshot_calls += 1
        return super().get_image(copy=copy)


def make_coordinator() -> tuple[
    CompletionRecognitionCoordinator,
    FakeSaveCoordinator,
    FakeRecognitionService,
]:
    save = FakeSaveCoordinator()
    recognition = FakeRecognitionService()
    coordinator = CompletionRecognitionCoordinator(
        save,  # type: ignore[arg-type]
        recognition,  # type: ignore[arg-type]
        CaseInputState(CharacterCaseMode.UPPERCASE),
    )
    return coordinator, save, recognition


def test_entering_done_saves_and_predicts_same_single_snapshot() -> None:
    coordinator, save, recognition = make_coordinator()
    canvas = CountingCanvas()
    canvas.draw_line((5, 5), (50, 40))

    assert coordinator.handle_transition(DrawingState.WRITING, canvas) is None
    result = coordinator.handle_transition(DrawingState.DONE, canvas)

    assert result is not None
    assert save.calls == 1
    assert recognition.calls == 1
    assert canvas.snapshot_calls == 1
    assert save.snapshots[0] is recognition.snapshots[0]
    assert result.prediction_result is not None
    assert result.prediction_result.prediction_id is not None


def test_remaining_done_does_not_repeat_work() -> None:
    coordinator, save, recognition = make_coordinator()
    canvas = CountingCanvas()
    canvas.draw_line((5, 5), (50, 40))

    coordinator.handle_transition(DrawingState.WRITING, canvas)
    coordinator.handle_transition(DrawingState.DONE, canvas)
    second = coordinator.handle_transition(DrawingState.DONE, canvas)

    assert second is None
    assert save.calls == recognition.calls == 1


def test_non_done_transitions_do_not_predict() -> None:
    coordinator, save, recognition = make_coordinator()
    canvas = CountingCanvas()

    for state in (DrawingState.READY, DrawingState.WRITING, DrawingState.PAUSED):
        assert coordinator.handle_transition(state, canvas) is None

    assert save.calls == recognition.calls == 0


def test_manual_prediction_does_not_save_or_change_state() -> None:
    coordinator, save, recognition = make_coordinator()
    canvas = CountingCanvas()
    canvas.draw_line((5, 5), (50, 40))

    result = coordinator.predict_now(canvas)

    assert result is not None
    assert result.status == PredictionStatus.ACCEPTED
    assert save.calls == 0
    assert recognition.calls == 1
    assert coordinator.previous_state == DrawingState.IDLE


def test_unavailable_model_returns_failure_but_still_saves() -> None:
    save = FakeSaveCoordinator()
    coordinator = CompletionRecognitionCoordinator(
        save,  # type: ignore[arg-type]
        None,
        CaseInputState(CharacterCaseMode.UPPERCASE),
    )
    canvas = CountingCanvas()
    canvas.draw_line((5, 5), (50, 40))

    result = coordinator.handle_transition(DrawingState.DONE, canvas)

    assert result is not None
    assert result.prediction_result is not None
    assert result.prediction_result.message == "Model unavailable"
    assert save.calls == 1


def test_pending_word_can_block_prediction_without_blocking_save() -> None:
    coordinator, save, recognition = make_coordinator()
    canvas = CountingCanvas()
    canvas.draw_line((5, 5), (50, 40))

    result = coordinator.handle_transition(
        DrawingState.DONE,
        canvas,
        allow_prediction=False,
    )

    assert result is not None
    assert result.save_result is not None
    assert result.prediction_result is None
    assert save.calls == 1
    assert recognition.calls == 0


def test_prediction_id_factory_is_used_for_distinct_events() -> None:
    identifiers = iter(("pred_1", "pred_2"))
    save = FakeSaveCoordinator()
    recognition = FakeRecognitionService()
    coordinator = CompletionRecognitionCoordinator(
        save,  # type: ignore[arg-type]
        recognition,  # type: ignore[arg-type]
        CaseInputState(CharacterCaseMode.UPPERCASE),
        prediction_id_factory=lambda: next(identifiers),
    )
    canvas = CountingCanvas()
    canvas.draw_line((5, 5), (50, 40))

    first = coordinator.predict_now(canvas)
    second = coordinator.predict_now(canvas)

    assert first is not None and first.prediction_id == "pred_1"
    assert second is not None and second.prediction_id == "pred_2"


def test_done_prediction_keeps_case_snapshot_after_global_mode_changes() -> None:
    save = FakeSaveCoordinator()
    recognition = FakeRecognitionService()
    case_state = CaseInputState(CharacterCaseMode.UPPERCASE)
    coordinator = CompletionRecognitionCoordinator(
        save,  # type: ignore[arg-type]
        recognition,  # type: ignore[arg-type]
        case_state,
    )
    canvas = CountingCanvas()
    canvas.draw_line((5, 5), (50, 40))

    coordinator.handle_transition(DrawingState.WRITING, canvas)
    completed = coordinator.handle_transition(DrawingState.DONE, canvas)
    case_state.set_lowercase()
    repeated = coordinator.handle_transition(DrawingState.DONE, canvas)

    assert completed is not None and completed.prediction_result is not None
    assert completed.prediction_result.case_selection.mode is CharacterCaseMode.UPPERCASE
    assert completed.prediction_result.top_prediction is not None
    assert completed.prediction_result.top_prediction.label == "A"
    assert repeated is None
    assert recognition.calls == 1
