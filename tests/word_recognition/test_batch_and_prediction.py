import numpy as np

from app.inference.case_label_resolver import CaseLabelResolver
from app.inference.character_predictor import CharacterPredictor
from app.inference.model_bundle import ModelBundle
from app.inference.prediction_policy import PredictionPolicy
from app.ml.letter_identity_labels import LETTER_IDENTITY_LABELS
from app.preprocessing.bounding_box import BoundingBox
from app.preprocessing.preprocessing_result import PreprocessingResult
from app.word_recognition.character_batch_builder import CharacterBatchBuilder
from app.word_recognition.character_segment import CharacterSegment
from app.word_recognition.hybrid_word_segmenter import HybridWordSegmenter
from app.word_recognition.isolated_letter_word_recognition_strategy import (
    IsolatedLetterWordRecognitionStrategy,
)
from app.word_recognition.whole_word_prediction_status import WholeWordPredictionStatus
from app.word_recognition.word_case_policy import WordCasePolicy
from app.word_recognition.word_input_snapshot import WordInputSnapshot
from app.word_recognition.word_writing_region import WordWritingRegion


class FakePreprocessor:
    def process(self, image: np.ndarray) -> PreprocessingResult:
        normalized = np.zeros((28, 28), dtype=np.float32)
        normalized[5:23, 8:20] = 1.0
        return PreprocessingResult(
            processed_image=(normalized * 255).astype(np.uint8),
            normalized_image=normalized,
            bounding_box=None,
            original_shape=image.shape,
            cropped_shape=image.shape,
            foreground_pixel_count=int(np.count_nonzero(image)),
        )


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0
        self.last_batch_shape: tuple[int, ...] | None = None

    def predict(self, images: np.ndarray, verbose: int) -> np.ndarray:
        self.calls += 1
        self.last_batch_shape = images.shape
        probabilities = np.full((images.shape[0], 26), 0.001, dtype=np.float32)
        for row in range(images.shape[0]):
            probabilities[row, row] = 0.90
            probabilities[row] /= probabilities[row].sum()
        return probabilities


def segment(position: int) -> CharacterSegment:
    return CharacterSegment(
        segment_id=f"s{position}",
        position=position,
        bounding_box=BoundingBox(position * 30, 0, 20, 30),
        source_stroke_ids=(),
        grayscale_image=np.full((30, 20), 255, dtype=np.uint8),
        segmentation_confidence=0.9,
    )


def predictor_and_model() -> tuple[CharacterPredictor, FakeModel]:
    model = FakeModel()
    bundle = ModelBundle(
        model=model,
        identity_labels=LETTER_IDENTITY_LABELS,
        lowercase_display_labels=LETTER_IDENTITY_LABELS,
        uppercase_display_labels=tuple(item.upper() for item in LETTER_IDENTITY_LABELS),
        model_version="test",
        task_type="letter_identity_classification",
        case_sensitive=False,
        case_source="user_selected_mode",
        expected_input_shape=(28, 28, 1),
        num_classes=26,
        preprocessing_contract={},
    )
    return CharacterPredictor(bundle), model


def test_batch_builder_preserves_segment_order_and_contract() -> None:
    batch = CharacterBatchBuilder(FakePreprocessor()).build((segment(0), segment(1)))

    assert batch.segment_ids == ("s0", "s1")
    assert batch.tensor.shape == (2, 28, 28, 1)
    assert batch.tensor.dtype == np.float32


def test_predict_batch_calls_model_once_for_all_characters() -> None:
    predictor, model = predictor_and_model()
    tensor = np.zeros((3, 28, 28, 1), dtype=np.float32)

    predictions = predictor.predict_batch(tensor)

    assert model.calls == 1
    assert model.last_batch_shape == (3, 28, 28, 1)
    assert [item.candidates[0].identity for item in predictions] == ["a", "b", "c"]


def test_strategy_segments_predicts_once_and_applies_capitalize_first() -> None:
    canvas = np.zeros((100, 180, 3), dtype=np.uint8)
    canvas[20:70, 10:35] = 255
    canvas[20:70, 55:80] = 255
    canvas[20:70, 100:125] = 255
    snapshot = WordInputSnapshot(
        "word",
        canvas,
        (),
        WordWritingRegion(0, 0, 180, 100),
        1,
    )
    predictor, model = predictor_and_model()
    resolver = CaseLabelResolver(
        LETTER_IDENTITY_LABELS,
        LETTER_IDENTITY_LABELS,
        tuple(item.upper() for item in LETTER_IDENTITY_LABELS),
    )
    strategy = IsolatedLetterWordRecognitionStrategy(
        HybridWordSegmenter(min_separator_gap=8),
        CharacterBatchBuilder(FakePreprocessor()),
        predictor,
        PredictionPolicy(0.60, 0.15),
        resolver,
    )

    result = strategy.recognize(snapshot, WordCasePolicy.CAPITALIZE_FIRST)

    assert result.status is WholeWordPredictionStatus.READY
    assert result.predicted_word == "Abc"
    assert model.calls == 1
