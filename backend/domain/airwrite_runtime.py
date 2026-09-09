from __future__ import annotations

import base64
import binascii
from time import time
from uuid import uuid4

import cv2
import numpy as np
from dataclasses import dataclass

from app.inference.character_predictor import CharacterPredictor
from app.inference.ensemble_character_predictor import EnsembleCharacterPredictor
from app.inference.identity_only_label_resolver import IdentityOnlyLabelResolver
from app.inference.model_bundle_loader import ModelBundleLoader
from app.inference.model_bundle_validator import ModelBundleValidator
from app.inference.prediction_policy import PredictionPolicy
from app.preprocessing.aspect_ratio_resizer import AspectRatioResizer
from app.preprocessing.bounding_box_extractor import BoundingBoxExtractor
from app.preprocessing.foreground_normalizer import ForegroundNormalizer
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor
from app.utils.config import identity_model_settings, inference_settings, settings, whole_word_settings
from app.word_recognition.character_batch_builder import CharacterBatchBuilder
from app.word_recognition.hybrid_word_segmenter import HybridWordSegmenter
from app.word_recognition.isolated_letter_word_recognition_strategy import IsolatedLetterWordRecognitionStrategy
from app.word_recognition.whole_word_recognition_service import WholeWordRecognitionService
from app.word_recognition.word_case_policy import WordCasePolicy
from app.word_recognition.word_input_snapshot import WordInputSnapshot
from app.word_recognition.word_writing_region import WordWritingRegion
from app.drawing.air_canvas import AirCanvas
from app.drawing.stroke_manager import StrokeManager
from app.vision.gesture_detector import GestureDetector
from app.vision.hand_detector import HandDetector
from app.vision.index_finger_tracker import IndexFingerTracker
from app.vision.gesture import Gesture
from app.vision.hand_landmark_renderer import HandLandmarkRenderer
from app.drawing.canvas_overlay_renderer import CanvasOverlayRenderer
from app.word_recognition.stroke_recorder import StrokeRecorder


@dataclass
class CaptureState:
    canvas: AirCanvas
    recorder: StrokeRecorder
    region: WordWritingRegion
    strokes: StrokeManager
    previous_point: tuple[int, int] | None = None
    stroke_start_canvas: np.ndarray | None = None
    clear_target_canvas: np.ndarray | None = None
    clear_target_stroke_count: int | None = None


class AirWriteRuntime:
    """Reusable local bridge to the existing AirWrite whole-word pipeline."""

    def __init__(self) -> None:
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
        bundle = ModelBundleLoader(
            model_path=identity_model_settings.model_path,
            identity_labels_path=identity_model_settings.identity_labels_path,
            lowercase_display_labels_path=identity_model_settings.lowercase_display_labels_path,
            uppercase_display_labels_path=identity_model_settings.uppercase_display_labels_path,
            metadata_path=identity_model_settings.metadata_path,
            preprocessing_config_path=identity_model_settings.preprocessing_config_path,
        ).load()
        ModelBundleValidator().validate(bundle, settings.preprocessing_runtime_contract())
        predictor = CharacterPredictor(bundle=bundle, top_k=inference_settings.top_k)
        emnist_root = identity_model_settings.emnist_model_path.parent
        try:
            emnist_bundle = ModelBundleLoader(
                model_path=identity_model_settings.emnist_model_path,
                identity_labels_path=emnist_root / "identity_labels.json",
                lowercase_display_labels_path=emnist_root / "lowercase_display_labels.json",
                uppercase_display_labels_path=emnist_root / "uppercase_display_labels.json",
                metadata_path=emnist_root / "model_metadata.json",
                preprocessing_config_path=emnist_root / "preprocessing_config.json",
            ).load()
            ModelBundleValidator().validate(emnist_bundle, settings.preprocessing_runtime_contract())
            predictor = EnsembleCharacterPredictor(
                custom=predictor,
                emnist=CharacterPredictor(emnist_bundle, top_k=inference_settings.top_k),
                custom_weight=0.80,
                top_k=inference_settings.top_k,
            )
        except (FileNotFoundError, RuntimeError, ValueError):
            # The custom AirWrite bundle remains a valid local fallback.
            pass
        resolver = IdentityOnlyLabelResolver(
            bundle.identity_labels,
            bundle.lowercase_display_labels,
            bundle.uppercase_display_labels,
        )
        strategy = IsolatedLetterWordRecognitionStrategy(
            segmenter=HybridWordSegmenter(
                min_foreground_pixels=whole_word_settings.min_foreground_pixels,
                min_component_area=whole_word_settings.min_component_area,
                min_separator_gap=whole_word_settings.min_separator_gap,
                min_characters=whole_word_settings.min_characters,
                max_characters=whole_word_settings.max_characters,
                wide_group_ratio=whole_word_settings.wide_group_ratio,
                x_overlap_threshold=whole_word_settings.x_overlap_threshold,
                tiny_component_ratio=whole_word_settings.tiny_component_ratio,
                max_internal_gap=whole_word_settings.max_internal_gap,
            ),
            batch_builder=CharacterBatchBuilder(preprocessor, whole_word_settings.max_characters),
            predictor=predictor,
            prediction_policy=PredictionPolicy(
                min_confidence=whole_word_settings.min_confidence,
                min_margin=whole_word_settings.min_margin,
            ),
            label_resolver=resolver,
        )
        self.service = WholeWordRecognitionService(strategy)
        self.detector = HandDetector(
            model_path=settings.hand_landmarker_model_path,
            num_hands=settings.hand_num_hands,
            min_detection_confidence=settings.hand_min_detection_confidence,
            min_presence_confidence=settings.hand_min_presence_confidence,
            min_tracking_confidence=settings.hand_min_tracking_confidence,
        )
        self.gesture_detector = GestureDetector(settings.finger_extension_margin)
        self.finger_tracker = IndexFingerTracker(
            settings.index_finger_landmark_index, settings.finger_smoothing_alpha
        )
        self.captures: dict[str, CaptureState] = {}
        self.landmark_renderer = HandLandmarkRenderer(
            settings.draw_hand_landmarks,
            settings.draw_hand_connections,
            settings.draw_handedness,
        )
        self.overlay_renderer = CanvasOverlayRenderer(
            settings.canvas_background_color,
            settings.canvas_overlay_opacity,
        )

    def process_camera_frame(self, session_id: str, encoded_image: str) -> dict[str, str | bool]:
        image = self._decode(encoded_image)
        if settings.camera_mirror:
            image = cv2.flip(image, 1)
        height, width = image.shape[:2]
        capture = self.captures.get(session_id)
        if capture is None or not capture.canvas.matches_size(width, height):
            region = WordWritingRegion.from_ratios(
                width, height,
                x_ratio=whole_word_settings.roi_x_ratio,
                y_ratio=whole_word_settings.roi_y_ratio,
                width_ratio=whole_word_settings.roi_width_ratio,
                height_ratio=whole_word_settings.roi_height_ratio,
            )
            capture = CaptureState(
                AirCanvas(width, height, settings.canvas_background_color, settings.canvas_stroke_color, settings.canvas_stroke_thickness),
                StrokeRecorder(whole_word_settings.max_strokes, whole_word_settings.max_points_per_stroke),
                region,
                StrokeManager(settings.canvas_max_point_distance),
            )
            self.captures[session_id] = capture
        detection = self.detector.detect(image, int(time() * 1000))
        gesture = self.gesture_detector.detect(detection)
        finger = self.finger_tracker.track(detection, width, height)
        point = finger.smoothed_point if gesture is Gesture.INDEX_ONLY else None
        if point is None or not capture.region.contains(point):
            capture.strokes.reset()
            capture.recorder.end_stroke()
            if capture.previous_point is not None and capture.stroke_start_canvas is not None:
                capture.clear_target_canvas = capture.stroke_start_canvas.copy()
                capture.clear_target_stroke_count = max(0, len(capture.recorder.completed_strokes) - 1)
            capture.previous_point = None
            capture.stroke_start_canvas = None
        else:
            if capture.previous_point is None:
                capture.stroke_start_canvas = capture.canvas.get_image(copy=True)
            segment = capture.strokes.update(point)
            if segment is not None:
                capture.canvas.draw_line(segment.start, segment.end)
            from app.word_recognition.stroke_point import StrokePoint
            if capture.recorder.is_recording:
                capture.recorder.append_point(StrokePoint(point[0], point[1], int(time() * 1000)))
            else:
                capture.recorder.start_stroke(StrokePoint(point[0], point[1], int(time() * 1000)))
            capture.previous_point = point
        annotated = self.landmark_renderer.draw(image, detection)
        annotated = self.overlay_renderer.render(annotated, capture.canvas.get_image())
        return {"state": gesture.value, "did_draw": point is not None, "frame": self._encode(annotated)}

    def clear_to_last_pause(self, session_id: str) -> dict[str, str]:
        capture = self.captures.get(session_id)
        if capture is None:
            raise ValueError("No camera capture is active")
        if capture.clear_target_canvas is not None:
            capture.canvas.get_image(copy=False)[:] = capture.clear_target_canvas
            if capture.clear_target_stroke_count is not None:
                capture.recorder.truncate(capture.clear_target_stroke_count)
            capture.clear_target_canvas = None
            capture.clear_target_stroke_count = None
        return {"status": "cleared"}

    def finish_camera_capture(self, session_id: str):
        capture = self.captures.get(session_id)
        if capture is None:
            raise ValueError("No camera capture is active")
        capture.recorder.end_stroke()
        return self.service.recognize(
            WordInputSnapshot(uuid4().hex, capture.canvas.get_image(), capture.recorder.snapshot(), capture.region, int(time() * 1000)),
            WordCasePolicy.LOWERCASE,
        )

    def recognize_canvas(self, encoded_image: str):
        image = self._decode(encoded_image)
        height, width = image.shape[:2]
        region = WordWritingRegion.from_ratios(
            width,
            height,
            x_ratio=whole_word_settings.roi_x_ratio,
            y_ratio=whole_word_settings.roi_y_ratio,
            width_ratio=whole_word_settings.roi_width_ratio,
            height_ratio=whole_word_settings.roi_height_ratio,
        )
        snapshot = WordInputSnapshot(
            snapshot_id=uuid4().hex,
            canvas_image=image,
            strokes=(),
            writing_region=region,
            completed_at_ms=int(time() * 1000),
        )
        return self.service.recognize(snapshot, WordCasePolicy.LOWERCASE)

    @staticmethod
    def _decode(encoded_image: str) -> np.ndarray:
        try:
            raw = base64.b64decode(encoded_image.split(",", 1)[-1], validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("Invalid camera image encoding") from error
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Camera image could not be decoded")
        return image

    @staticmethod
    def _encode(image: np.ndarray) -> str:
        success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not success:
            raise ValueError("Unable to encode AirWrite canvas")
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")
