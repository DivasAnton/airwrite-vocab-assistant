import argparse
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.inference.case_label_resolver import CaseLabelResolver
from app.inference.case_selection import CaseSelection
from app.inference.character_case_mode import CharacterCaseMode
from app.inference.character_predictor import CharacterPredictor
from app.inference.character_recognition_service import CharacterRecognitionService
from app.inference.model_bundle_loader import ModelBundleLoader
from app.inference.model_bundle_validator import ModelBundleValidator
from app.inference.prediction_policy import PredictionPolicy
from app.preprocessing.aspect_ratio_resizer import AspectRatioResizer
from app.preprocessing.bounding_box_extractor import BoundingBoxExtractor
from app.preprocessing.foreground_normalizer import ForegroundNormalizer
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor
from app.utils.config import identity_model_settings, inference_settings, settings


def build_preprocessor() -> HandwritingPreprocessor:
    return HandwritingPreprocessor(
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run case-controlled identity inference on an image"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mode", choices=("lowercase", "uppercase"), required=True)
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {args.image}")
    image = cast(NDArray[np.uint8], image)

    bundle = ModelBundleLoader(
        model_path=identity_model_settings.model_path,
        identity_labels_path=identity_model_settings.identity_labels_path,
        lowercase_display_labels_path=identity_model_settings.lowercase_display_labels_path,
        uppercase_display_labels_path=identity_model_settings.uppercase_display_labels_path,
        metadata_path=identity_model_settings.metadata_path,
        preprocessing_config_path=identity_model_settings.preprocessing_config_path,
    ).load()
    ModelBundleValidator().validate(bundle, settings.preprocessing_runtime_contract())
    resolver = CaseLabelResolver(
        bundle.identity_labels,
        bundle.lowercase_display_labels,
        bundle.uppercase_display_labels,
    )
    service = CharacterRecognitionService(
        preprocessor=build_preprocessor(),
        predictor=CharacterPredictor(bundle, top_k=inference_settings.top_k),
        policy=PredictionPolicy(
            min_confidence=inference_settings.min_confidence,
            min_margin=inference_settings.min_margin,
        ),
        case_resolver=resolver,
        log_latency=False,
    )
    mode = CharacterCaseMode.from_string(args.mode)
    result = service.recognize(
        image,
        case_selection=CaseSelection(mode=mode, shift_was_active=False),
    )

    print(f"Status: {result.status.value}")
    if result.top_prediction is None:
        print(result.message)
        return
    print(f"Identity: {result.top_prediction.identity}")
    print(f"Rendered character: {result.top_prediction.rendered_character}")
    print(
        "Top-3: "
        + ", ".join(
            f"{candidate.rendered_character} ({candidate.confidence:.2%})"
            for candidate in result.candidates
        )
    )
    print("Case inferred by model: no")


if __name__ == "__main__":
    main()
