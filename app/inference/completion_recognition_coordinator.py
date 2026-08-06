from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from app.drawing.air_canvas import AirCanvas
from app.drawing.drawing_state import DrawingState
from app.inference.case_input_state import CaseInputState
from app.inference.case_selection import CaseSelection
from app.inference.character_recognition_service import CharacterRecognitionService
from app.inference.prediction_result import PredictionResult
from app.inference.prediction_status import PredictionStatus
from app.storage.drawing_save_coordinator import DrawingSaveCoordinator
from app.storage.save_result import SaveResult


@dataclass(frozen=True)
class CompletionRecognitionResult:
    save_result: SaveResult | None
    prediction_result: PredictionResult | None


class CompletionRecognitionCoordinator:
    def __init__(
        self,
        save_coordinator: DrawingSaveCoordinator,
        recognition_service: CharacterRecognitionService | None,
        case_state: CaseInputState,
        auto_predict_on_done: bool = True,
        manual_predict_enabled: bool = True,
        unavailable_model_version: str | None = None,
        prediction_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.save_coordinator = save_coordinator
        self.recognition_service = recognition_service
        self.case_state = case_state
        self.auto_predict_on_done = auto_predict_on_done
        self.manual_predict_enabled = manual_predict_enabled
        self.unavailable_model_version = unavailable_model_version
        self.prediction_id_factory = prediction_id_factory or self._default_prediction_id
        self.previous_state = DrawingState.IDLE

    def handle_transition(
        self,
        current_state: DrawingState,
        canvas: AirCanvas,
        allow_prediction: bool = True,
    ) -> CompletionRecognitionResult | None:
        entered_done = (
            self.previous_state != DrawingState.DONE and current_state == DrawingState.DONE
        )
        self.previous_state = current_state
        if not entered_done:
            return None

        is_empty = canvas.is_empty()
        snapshot = canvas.get_image(copy=True)
        case_selection = self.case_state.create_selection()
        save_result = self.save_coordinator.save_snapshot(
            snapshot,
            is_empty=is_empty,
            manual=False,
        )
        prediction_result = None
        if self.auto_predict_on_done and allow_prediction:
            prediction_result = self._recognize(
                snapshot,
                is_empty=is_empty,
                case_selection=case_selection,
            )
        return CompletionRecognitionResult(
            save_result=save_result,
            prediction_result=prediction_result,
        )

    def predict_now(self, canvas: AirCanvas) -> PredictionResult | None:
        if not self.manual_predict_enabled:
            return None
        is_empty = canvas.is_empty()
        snapshot = canvas.get_image(copy=True)
        case_selection = self.case_state.create_selection()
        return self._recognize(
            snapshot,
            is_empty=is_empty,
            case_selection=case_selection,
        )

    def _recognize(
        self,
        snapshot: NDArray[np.uint8],
        *,
        is_empty: bool,
        case_selection: CaseSelection,
    ) -> PredictionResult:
        prediction_id = self._next_prediction_id()
        if is_empty:
            return PredictionResult(
                prediction_id=prediction_id,
                status=PredictionStatus.SKIPPED_EMPTY,
                top_prediction=None,
                candidates=(),
                confidence_margin=None,
                inference_time_ms=None,
                model_version=self.unavailable_model_version,
                message="Nothing to predict",
                case_selection=case_selection,
                case_was_inferred=False,
            )
        if self.recognition_service is None:
            return PredictionResult(
                prediction_id=prediction_id,
                status=PredictionStatus.FAILED,
                top_prediction=None,
                candidates=(),
                confidence_margin=None,
                inference_time_ms=None,
                model_version=self.unavailable_model_version,
                message="Model unavailable",
                case_selection=case_selection,
                case_was_inferred=False,
            )
        return self.recognition_service.recognize(
            snapshot,
            case_selection=case_selection,
            prediction_id=prediction_id,
        )

    def _next_prediction_id(self) -> str:
        prediction_id = self.prediction_id_factory()
        if not prediction_id.strip():
            raise ValueError("prediction_id_factory must return a non-empty identifier")
        return prediction_id

    @staticmethod
    def _default_prediction_id() -> str:
        return f"pred_{uuid4().hex}"
