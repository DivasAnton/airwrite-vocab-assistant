import argparse
from pathlib import Path

import cv2

from app.inference.case_label_resolver import CaseLabelResolver
from app.inference.character_predictor import CharacterPredictor
from app.inference.model_bundle_loader import ModelBundleLoader
from app.inference.model_bundle_validator import ModelBundleValidator
from app.inference.prediction_policy import PredictionPolicy
from app.preprocessing.aspect_ratio_resizer import AspectRatioResizer
from app.preprocessing.bounding_box_extractor import BoundingBoxExtractor
from app.preprocessing.foreground_normalizer import ForegroundNormalizer
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor
from app.utils.config import identity_model_settings, settings, whole_word_settings
from app.word_recognition.character_batch_builder import CharacterBatchBuilder
from app.word_recognition.hybrid_word_segmenter import HybridWordSegmenter
from app.word_recognition.isolated_letter_word_recognition_strategy import (
    IsolatedLetterWordRecognitionStrategy,
)
from app.word_recognition.word_case_policy import WordCasePolicy
from app.word_recognition.word_input_snapshot import WordInputSnapshot
from app.word_recognition.word_writing_region import WordWritingRegion


def build_strategy() -> IsolatedLetterWordRecognitionStrategy:
    bundle = ModelBundleLoader(
        model_path=identity_model_settings.model_path,
        identity_labels_path=identity_model_settings.identity_labels_path,
        lowercase_display_labels_path=identity_model_settings.lowercase_display_labels_path,
        uppercase_display_labels_path=identity_model_settings.uppercase_display_labels_path,
        metadata_path=identity_model_settings.metadata_path,
        preprocessing_config_path=identity_model_settings.preprocessing_config_path,
    ).load()
    ModelBundleValidator().validate(bundle, settings.preprocessing_runtime_contract())
    preprocessor = HandwritingPreprocessor(
        normalizer=ForegroundNormalizer(
            binary_threshold=settings.preprocess_binary_threshold,
            invert_input=settings.preprocess_invert_input,
        ),
        extractor=BoundingBoxExtractor(
            crop_padding=settings.preprocess_crop_padding,
            min_foreground_pixels=settings.preprocess_min_foreground_pixels,
        ),
        resizer=AspectRatioResizer(
            output_width=settings.preprocess_output_width,
            output_height=settings.preprocess_output_height,
            content_width=settings.preprocess_content_width,
            content_height=settings.preprocess_content_height,
            center_of_mass=settings.preprocess_center_of_mass,
        ),
    )
    segmenter = HybridWordSegmenter(
        min_foreground_pixels=whole_word_settings.min_foreground_pixels,
        min_component_area=whole_word_settings.min_component_area,
        min_separator_gap=whole_word_settings.min_separator_gap,
        min_characters=whole_word_settings.min_characters,
        max_characters=whole_word_settings.max_characters,
        wide_group_ratio=whole_word_settings.wide_group_ratio,
        x_overlap_threshold=whole_word_settings.x_overlap_threshold,
        tiny_component_ratio=whole_word_settings.tiny_component_ratio,
        max_internal_gap=whole_word_settings.max_internal_gap,
    )
    return IsolatedLetterWordRecognitionStrategy(
        segmenter,
        CharacterBatchBuilder(preprocessor, whole_word_settings.max_characters),
        CharacterPredictor(bundle, whole_word_settings.top_k),
        PredictionPolicy(
            whole_word_settings.min_confidence,
            whole_word_settings.min_margin,
        ),
        CaseLabelResolver(
            bundle.identity_labels,
            bundle.lowercase_display_labels,
            bundle.uppercase_display_labels,
        ),
    )


def recognize_image(
    strategy: IsolatedLetterWordRecognitionStrategy,
    image_path: Path,
    policy: WordCasePolicy,
) -> tuple[str, float, str]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    result = strategy.recognize(
        WordInputSnapshot(
            image_path.stem,
            image,
            (),
            WordWritingRegion(0, 0, width, height),
            0,
        ),
        policy,
    )
    return result.predicted_word, result.total_time_ms, result.status.value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run whole-word recognition on a saved canvas")
    parser.add_argument("image", type=Path)
    parser.add_argument("--expected")
    parser.add_argument("--case-policy", default="lowercase")
    args = parser.parse_args()
    predicted, elapsed_ms, status = recognize_image(
        build_strategy(),
        args.image,
        WordCasePolicy.from_string(args.case_policy),
    )
    print(f"status={status} predicted={predicted!r} total_ms={elapsed_ms:.2f}")
    if args.expected is not None and predicted != args.expected:
        raise SystemExit(f"Expected {args.expected!r}, got {predicted!r}")


if __name__ == "__main__":
    main()
