from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.drawing.air_canvas import AirCanvas
from app.drawing.drawing_state import DrawingState
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
        auto_predict_on_done: bool = True,
        manual_predict_enabled: bool = True,
        unavailable_model_version: str | None = None,
    ) -> None:
        self.save_coordinator = save_coordinator
        self.recognition_service = recognition_service
        self.auto_predict_on_done = auto_predict_on_done
        self.manual_predict_enabled = manual_predict_enabled
        self.unavailable_model_version = unavailable_model_version
        self.previous_state = DrawingState.IDLE

    def handle_transition(
        self,
        current_state: DrawingState,
        canvas: AirCanvas,
    ) -> CompletionRecognitionResult | None:
        entered_done = (
            self.previous_state != DrawingState.DONE and current_state == DrawingState.DONE
        )
        self.previous_state = current_state
        if not entered_done:
            return None

        is_empty = canvas.is_empty()
        snapshot = canvas.get_image(copy=True)
        save_result = self.save_coordinator.save_snapshot(
            snapshot,
            is_empty=is_empty,
            manual=False,
        )
        prediction_result = (
            self._recognize(snapshot, is_empty=is_empty) if self.auto_predict_on_done else None
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
        return self._recognize(snapshot, is_empty=is_empty)

    def _recognize(
        self,
        snapshot: NDArray[np.uint8],
        *,
        is_empty: bool,
    ) -> PredictionResult:
        if is_empty:
            return PredictionResult(
                status=PredictionStatus.SKIPPED_EMPTY,
                top_prediction=None,
                candidates=(),
                confidence_margin=None,
                inference_time_ms=None,
                model_version=self.unavailable_model_version,
                message="Nothing to predict",
            )
        if self.recognition_service is None:
            return PredictionResult(
                status=PredictionStatus.FAILED,
                top_prediction=None,
                candidates=(),
                confidence_margin=None,
                inference_time_ms=None,
                model_version=self.unavailable_model_version,
                message="Model unavailable",
            )
        return self.recognition_service.recognize(snapshot)
