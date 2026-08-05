import logging

import numpy as np
from numpy.typing import NDArray

from app.inference.character_predictor import CharacterPredictor
from app.inference.exceptions import InferenceError
from app.inference.prediction_policy import PredictionPolicy
from app.inference.prediction_result import PredictionResult
from app.inference.prediction_status import PredictionStatus
from app.preprocessing.exceptions import EmptyDrawingError, PreprocessingError
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor


class CharacterRecognitionService:
    def __init__(
        self,
        preprocessor: HandwritingPreprocessor,
        predictor: CharacterPredictor,
        policy: PredictionPolicy,
        logger: logging.Logger | None = None,
        log_latency: bool = True,
        latency_warning_ms: int = 500,
    ) -> None:
        if latency_warning_ms <= 0:
            raise ValueError("latency_warning_ms must be greater than 0")
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.policy = policy
        self.logger = logger or logging.getLogger(__name__)
        self.log_latency = log_latency
        self.latency_warning_ms = latency_warning_ms

    def recognize(self, canvas_snapshot: NDArray[np.uint8]) -> PredictionResult:
        try:
            preprocessing_result = self.preprocessor.process(canvas_snapshot.copy())
            raw_prediction = self.predictor.predict(preprocessing_result.normalized_image)
            result = self.policy.apply(raw_prediction, self.predictor.model_version)
        except EmptyDrawingError:
            return PredictionResult(
                status=PredictionStatus.SKIPPED_EMPTY,
                top_prediction=None,
                candidates=(),
                confidence_margin=None,
                inference_time_ms=None,
                model_version=self.predictor.model_version,
                message="Nothing to predict",
            )
        except (PreprocessingError, InferenceError) as error:
            self.logger.exception("Character recognition failed: %s", error)
            return self._failed_result()

        self._log_latency(result)
        return result

    def _log_latency(self, result: PredictionResult) -> None:
        if not self.log_latency or result.inference_time_ms is None:
            return
        self.logger.info("Prediction completed in %.1f ms", result.inference_time_ms)
        if result.inference_time_ms > self.latency_warning_ms:
            self.logger.warning("Prediction latency exceeded expected prototype threshold")

    def _failed_result(self) -> PredictionResult:
        return PredictionResult(
            status=PredictionStatus.FAILED,
            top_prediction=None,
            candidates=(),
            confidence_margin=None,
            inference_time_ms=None,
            model_version=self.predictor.model_version,
            message="Prediction failed",
        )
