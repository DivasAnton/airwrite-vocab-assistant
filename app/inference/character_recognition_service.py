import logging
from collections.abc import Callable
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from app.inference.case_label_resolver import CaseLabelResolver
from app.inference.case_selection import CaseSelection
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
        case_resolver: CaseLabelResolver,
        logger: logging.Logger | None = None,
        log_latency: bool = True,
        latency_warning_ms: int = 500,
        prediction_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if latency_warning_ms <= 0:
            raise ValueError("latency_warning_ms must be greater than 0")
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.policy = policy
        self.case_resolver = case_resolver
        self.logger = logger or logging.getLogger(__name__)
        self.log_latency = log_latency
        self.latency_warning_ms = latency_warning_ms
        self.prediction_id_factory = prediction_id_factory or self._default_prediction_id

    def recognize(
        self,
        canvas_snapshot: NDArray[np.uint8],
        *,
        case_selection: CaseSelection,
        prediction_id: str | None = None,
    ) -> PredictionResult:
        event_id = prediction_id or self.prediction_id_factory()
        if not event_id.strip():
            raise ValueError("prediction_id must not be empty")
        try:
            preprocessing_result = self.preprocessor.process(canvas_snapshot.copy())
            raw_prediction = self.predictor.predict(preprocessing_result.normalized_image)
            status, margin = self.policy.evaluate(raw_prediction.candidates)
            candidates = tuple(
                self.case_resolver.resolve_candidate(candidate, case_selection.mode)
                for candidate in raw_prediction.candidates
            )
            top_prediction = candidates[0]
            message = (
                f"Prediction accepted: {top_prediction.rendered_character}"
                if status == PredictionStatus.ACCEPTED
                else "Prediction is uncertain"
            )
            result = PredictionResult(
                prediction_id=event_id,
                status=status,
                top_prediction=top_prediction,
                candidates=candidates,
                confidence_margin=margin,
                inference_time_ms=raw_prediction.inference_time_ms,
                model_version=self.predictor.model_version,
                message=message,
                case_selection=case_selection,
                case_was_inferred=False,
            )
        except EmptyDrawingError:
            return PredictionResult(
                prediction_id=event_id,
                status=PredictionStatus.SKIPPED_EMPTY,
                top_prediction=None,
                candidates=(),
                confidence_margin=None,
                inference_time_ms=None,
                model_version=self.predictor.model_version,
                message="Nothing to predict",
                case_selection=case_selection,
                case_was_inferred=False,
            )
        except (PreprocessingError, InferenceError) as error:
            self.logger.exception("Character recognition failed: %s", error)
            return self._failed_result(event_id, case_selection)

        self._log_latency(result)
        return result

    def _log_latency(self, result: PredictionResult) -> None:
        if not self.log_latency or result.inference_time_ms is None:
            return
        self.logger.info("Prediction completed in %.1f ms", result.inference_time_ms)
        if result.inference_time_ms > self.latency_warning_ms:
            self.logger.warning("Prediction latency exceeded expected prototype threshold")

    def _failed_result(self, prediction_id: str, case_selection: CaseSelection) -> PredictionResult:
        return PredictionResult(
            prediction_id=prediction_id,
            status=PredictionStatus.FAILED,
            top_prediction=None,
            candidates=(),
            confidence_margin=None,
            inference_time_ms=None,
            model_version=self.predictor.model_version,
            message="Prediction failed",
            case_selection=case_selection,
            case_was_inferred=False,
        )

    @staticmethod
    def _default_prediction_id() -> str:
        return f"pred_{uuid4().hex}"
