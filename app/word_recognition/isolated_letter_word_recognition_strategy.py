from collections.abc import Mapping
from time import perf_counter

from app.inference.case_label_resolver import CaseLabelResolver
from app.inference.character_case_mode import CharacterCaseMode
from app.inference.character_predictor import CharacterPredictor, RawPrediction
from app.inference.prediction_policy import PredictionPolicy
from app.inference.prediction_status import PredictionStatus
from app.word_builder.character_source import CharacterSource
from app.word_recognition.character_batch_builder import CharacterBatchBuilder
from app.word_recognition.character_segment import CharacterSegment
from app.word_recognition.hybrid_word_segmenter import HybridWordSegmenter
from app.word_recognition.segmentation_status import SegmentationStatus
from app.word_recognition.whole_word_prediction_result import WholeWordPredictionResult
from app.word_recognition.whole_word_prediction_status import WholeWordPredictionStatus
from app.word_recognition.word_case_policy import WordCasePolicy
from app.word_recognition.word_case_resolver import WordCaseResolver
from app.word_recognition.word_input_snapshot import WordInputSnapshot
from app.word_recognition.word_prediction_character import WordPredictionCharacter


class IsolatedLetterWordRecognitionStrategy:
    def __init__(
        self,
        segmenter: HybridWordSegmenter,
        batch_builder: CharacterBatchBuilder,
        predictor: CharacterPredictor,
        prediction_policy: PredictionPolicy,
        label_resolver: CaseLabelResolver,
        case_resolver: WordCaseResolver | None = None,
    ) -> None:
        self.segmenter = segmenter
        self.batch_builder = batch_builder
        self.predictor = predictor
        self.prediction_policy = prediction_policy
        self.label_resolver = label_resolver
        self.case_resolver = case_resolver or WordCaseResolver()

    def recognize(
        self,
        snapshot: WordInputSnapshot,
        case_policy: WordCasePolicy,
    ) -> WholeWordPredictionResult:
        started_at = perf_counter()
        segmentation = self.segmenter.segment(snapshot)
        if not segmentation.segments:
            status = (
                WholeWordPredictionStatus.EMPTY
                if segmentation.status is SegmentationStatus.EMPTY
                else WholeWordPredictionStatus.SEGMENTATION_FAILED
            )
            return self._result(
                snapshot.snapshot_id,
                status,
                (),
                segmentation.status,
                (),
                segmentation.elapsed_ms,
                0.0,
                0.0,
                started_at,
                segmentation.message,
            )

        preprocessing_started = perf_counter()
        try:
            batch = self.batch_builder.build(segmentation.segments)
        except (TypeError, ValueError, RuntimeError) as error:
            return self._result(
                snapshot.snapshot_id,
                WholeWordPredictionStatus.INFERENCE_FAILED,
                (),
                segmentation.status,
                segmentation.segments,
                segmentation.elapsed_ms,
                (perf_counter() - preprocessing_started) * 1000.0,
                0.0,
                started_at,
                f"Character preprocessing failed: {error}",
            )
        preprocessing_ms = (perf_counter() - preprocessing_started) * 1000.0
        try:
            predictions = self.predictor.predict_batch(batch.tensor)
        except (TypeError, ValueError, RuntimeError) as error:
            return self._result(
                snapshot.snapshot_id,
                WholeWordPredictionStatus.INFERENCE_FAILED,
                (),
                segmentation.status,
                segmentation.segments,
                segmentation.elapsed_ms,
                preprocessing_ms,
                0.0,
                started_at,
                f"Whole-word prediction failed: {error}",
            )
        characters = self._characters(
            segmentation.segments,
            predictions,
            case_policy,
        )
        inference_ms = max(
            (prediction.inference_time_ms for prediction in predictions), default=0.0
        )
        needs_review = segmentation.status is not SegmentationStatus.SUCCESS or any(
            character.status is PredictionStatus.UNCERTAIN for character in characters
        )
        status = (
            WholeWordPredictionStatus.NEEDS_REVIEW
            if needs_review
            else WholeWordPredictionStatus.READY
        )
        return self._result(
            snapshot.snapshot_id,
            status,
            characters,
            segmentation.status,
            segmentation.segments,
            segmentation.elapsed_ms,
            preprocessing_ms,
            inference_ms,
            started_at,
            "Review the whole-word draft" if needs_review else "Whole-word draft is ready",
        )

    def predict_segments(
        self,
        prediction_id: str,
        segments: tuple[CharacterSegment, ...],
        case_policy: WordCasePolicy,
        overrides: Mapping[int, CharacterCaseMode] | None = None,
    ) -> tuple[WordPredictionCharacter, ...]:
        normalized = tuple(
            CharacterSegment(
                segment_id=segment.segment_id,
                position=position,
                bounding_box=segment.bounding_box,
                source_stroke_ids=segment.source_stroke_ids,
                grayscale_image=segment.grayscale_image,
                segmentation_confidence=segment.segmentation_confidence,
            )
            for position, segment in enumerate(segments)
        )
        batch = self.batch_builder.build(normalized)
        return self._characters(
            normalized,
            self.predictor.predict_batch(batch.tensor),
            case_policy,
            overrides,
        )

    def _characters(
        self,
        segments: tuple[CharacterSegment, ...],
        predictions: tuple[RawPrediction, ...],
        case_policy: WordCasePolicy,
        overrides: Mapping[int, CharacterCaseMode] | None = None,
    ) -> tuple[WordPredictionCharacter, ...]:
        modes = self.case_resolver.resolve_modes(len(segments), case_policy, overrides)
        characters: list[WordPredictionCharacter] = []
        for position, (segment, prediction, mode) in enumerate(
            zip(segments, predictions, modes, strict=True)
        ):
            status, _margin = self.prediction_policy.evaluate(prediction.candidates)
            candidates = tuple(
                self.label_resolver.resolve_candidate(candidate, mode)
                for candidate in prediction.candidates
            )
            top = candidates[0]
            characters.append(
                WordPredictionCharacter(
                    position=position,
                    identity=top.identity,
                    rendered_character=top.rendered_character,
                    case_mode=mode,
                    confidence=top.confidence,
                    candidates=candidates,
                    status=status,
                    source=(
                        CharacterSource.AUTO_ACCEPTED
                        if status is PredictionStatus.ACCEPTED
                        else CharacterSource.USER_SELECTED
                    ),
                    segment_id=segment.segment_id,
                    bounding_box=segment.bounding_box,
                )
            )
        return tuple(characters)

    @staticmethod
    def _result(
        prediction_id: str,
        status: WholeWordPredictionStatus,
        characters: tuple[WordPredictionCharacter, ...],
        segmentation_status: SegmentationStatus,
        segments: tuple[CharacterSegment, ...],
        segmentation_ms: float,
        preprocessing_ms: float,
        inference_ms: float,
        started_at: float,
        message: str,
    ) -> WholeWordPredictionResult:
        return WholeWordPredictionResult(
            prediction_id=prediction_id,
            status=status,
            characters=characters,
            predicted_word="".join(item.rendered_character for item in characters),
            segmentation_time_ms=segmentation_ms,
            preprocessing_time_ms=preprocessing_ms,
            inference_time_ms=inference_ms,
            total_time_ms=(perf_counter() - started_at) * 1000.0,
            segmentation_status=segmentation_status,
            segments=segments,
            message=message,
        )
