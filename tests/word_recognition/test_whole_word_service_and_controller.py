import numpy as np
from app.inference.prediction_status import PredictionStatus
from app.word_builder.word_action import WordAction
from app.word_builder.word_builder import WordBuilder
from app.word_recognition.whole_word_controller import WholeWordController
from app.word_recognition.whole_word_recognition_service import WholeWordRecognitionService
from app.word_recognition.word_case_policy import WordCasePolicy
from app.word_recognition.word_input_snapshot import WordInputSnapshot
from app.word_recognition.word_writing_region import WordWritingRegion
from tests.word_recognition.test_draft_and_atomic_commit import (
    prediction_character,
    result,
)


class FakeStrategy:
    def __init__(self, prediction_result: object) -> None:
        self.prediction_result = prediction_result
        self.calls = 0

    def recognize(self, snapshot: object, case_policy: object) -> object:
        self.calls += 1
        return self.prediction_result


def snapshot() -> WordInputSnapshot:
    return WordInputSnapshot(
        "snapshot",
        np.zeros((50, 100, 3), dtype=np.uint8),
        (),
        WordWritingRegion(0, 0, 100, 50),
        1,
    )


def test_controller_does_not_commit_unresolved_draft() -> None:
    prediction = result(
        (
            prediction_character(0, "c"),
            prediction_character(1, "a", PredictionStatus.UNCERTAIN),
            prediction_character(2, "t"),
        )
    )
    strategy = FakeStrategy(prediction)
    builder = WordBuilder()
    controller = WholeWordController(
        WholeWordRecognitionService(strategy),  # type: ignore[arg-type]
        strategy,  # type: ignore[arg-type]
        builder,
    )

    controller.recognize(snapshot(), WordCasePolicy.LOWERCASE)
    rejected = controller.accept()

    assert rejected.action is WordAction.ERROR
    assert builder.current_word == ""


def test_controller_accepts_resolved_word_as_one_atomic_action() -> None:
    prediction = result(
        (
            prediction_character(0, "c"),
            prediction_character(1, "a"),
            prediction_character(2, "t"),
        )
    )
    strategy = FakeStrategy(prediction)
    builder = WordBuilder()
    controller = WholeWordController(
        WholeWordRecognitionService(strategy),  # type: ignore[arg-type]
        strategy,  # type: ignore[arg-type]
        builder,
    )

    controller.recognize(snapshot(), WordCasePolicy.LOWERCASE)
    committed = controller.accept()

    assert committed.action is WordAction.WHOLE_WORD_COMMITTED
    assert committed.current_word == "cat"
    assert controller.draft is None
