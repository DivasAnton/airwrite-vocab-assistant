from time import perf_counter
from typing import cast
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

from app.camera.camera_stream import CameraStream
from app.camera.frame_processor import FrameProcessor
from app.drawing.air_canvas import AirCanvas
from app.drawing.canvas_overlay_renderer import CanvasOverlayRenderer
from app.drawing.drawing_controller import DrawingController
from app.drawing.drawing_state import DrawingState
from app.drawing.drawing_state_machine import DrawingStateMachine
from app.drawing.stroke_manager import StrokeManager
from app.inference.case_input_state import CaseInputState
from app.inference.case_label_resolver import CaseLabelResolver
from app.inference.case_mode_controller import CaseAction, CaseModeController
from app.inference.character_predictor import CharacterPredictor
from app.inference.character_recognition_service import CharacterRecognitionService
from app.inference.completion_recognition_coordinator import (
    CompletionRecognitionCoordinator,
)
from app.inference.ensemble_character_predictor import EnsembleCharacterPredictor
from app.inference.exceptions import InferenceError
from app.inference.identity_only_label_resolver import IdentityOnlyLabelResolver
from app.inference.model_bundle_loader import ModelBundleLoader
from app.inference.model_bundle_validator import ModelBundleValidator
from app.inference.prediction_policy import PredictionPolicy
from app.inference.prediction_result import PredictionResult
from app.inference.prediction_status_renderer import PredictionStatusRenderer
from app.preprocessing.aspect_ratio_resizer import AspectRatioResizer
from app.preprocessing.bounding_box_extractor import BoundingBoxExtractor
from app.preprocessing.debug_image_exporter import DebugImageExporter
from app.preprocessing.exceptions import EmptyDrawingError, PreprocessingError
from app.preprocessing.foreground_normalizer import ForegroundNormalizer
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor
from app.preprocessing.preprocessing_result import PreprocessingResult
from app.status_hud_renderer import StatusHudRenderer
from app.storage.character_dataset_saver import CharacterDatasetImageSaver
from app.storage.drawing_image_saver import DrawingImageSaver
from app.storage.drawing_save_coordinator import DrawingSaveCoordinator
from app.storage.exceptions import DrawingStorageError
from app.storage.save_result import SaveResult, SaveStatus
from app.utils.config import (
    case_control_settings,
    identity_model_settings,
    inference_settings,
    settings,
    whole_word_settings,
    word_builder_settings,
)
from app.utils.logger import get_logger
from app.vision.finger_tracking_renderer import FingerTrackingRenderer
from app.vision.gesture import Gesture
from app.vision.gesture_detector import GestureDetector
from app.vision.gesture_stabilizer import GestureStabilizer
from app.vision.hand_detector import HandDetector
from app.vision.hand_landmark_renderer import HandLandmarkRenderer
from app.vision.index_finger_tracker import IndexFingerTracker
from app.word_builder.case_state_coordinator import update_case_state_after_word_action
from app.word_builder.supported_character_set import SupportedCharacterSet
from app.word_builder.word_action import WordAction
from app.word_builder.word_builder import WordBuilder
from app.word_builder.word_builder_controller import WordBuilderController
from app.word_builder.word_builder_result import WordBuilderResult
from app.word_recognition.character_batch_builder import CharacterBatchBuilder
from app.word_recognition.drawing_input_mode import DrawingInputMode
from app.word_recognition.hybrid_word_segmenter import HybridWordSegmenter
from app.word_recognition.input_mode_controller import InputModeController
from app.word_recognition.isolated_letter_word_recognition_strategy import (
    IsolatedLetterWordRecognitionStrategy,
)
from app.word_recognition.stroke_recorder import StrokeRecorder
from app.word_recognition.whole_word_controller import WholeWordController
from app.word_recognition.whole_word_recognition_service import WholeWordRecognitionService
from app.word_recognition.whole_word_renderer import WholeWordRenderer
from app.word_recognition.word_case_policy import WordCasePolicy
from app.word_recognition.word_input_snapshot import WordInputSnapshot
from app.word_recognition.word_writing_region import WordWritingRegion

logger = get_logger(__name__)


def should_quit(window_name: str, key: int) -> bool:
    if key in {27, ord("q"), ord("Q")}:
        return True

    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def next_timestamp_ms(previous_timestamp_ms: int) -> int:
    timestamp_ms = int(perf_counter() * 1000)
    return max(timestamp_ms, previous_timestamp_ms + 1)


def is_clear_key(key: int, clear_key: str) -> bool:
    return key in {ord(clear_key.lower()), ord(clear_key.upper())}


def is_done_key(key: int) -> bool:
    return key in {ord("d"), ord("D")}


def is_manual_save_key(key: int, manual_save_key: str) -> bool:
    return key in {ord(manual_save_key.lower()), ord(manual_save_key.upper())}


def is_manual_preprocess_key(key: int, manual_preprocess_key: str) -> bool:
    return key in {ord(manual_preprocess_key.lower()), ord(manual_preprocess_key.upper())}


def is_manual_predict_key(key: int, manual_predict_key: str) -> bool:
    return key in {ord(manual_predict_key.lower()), ord(manual_predict_key.upper())}


def is_dataset_capture_key(key: int, capture_key: str) -> bool:
    return key in {ord(capture_key.lower()), ord(capture_key.upper())}


def is_dataset_label_next_key(key: int, next_key: str) -> bool:
    return key == ord(next_key)


def is_dataset_label_previous_key(key: int, previous_key: str) -> bool:
    return key == ord(previous_key)


def is_space_key(key: int) -> bool:
    return key == ord(" ")


def is_named_key(key: int, configured_key: str) -> bool:
    return key in {ord(configured_key.lower()), ord(configured_key.upper())}


def case_action_for_key(key: int) -> CaseAction | None:
    mappings = (
        (case_control_settings.lowercase_mode_key, CaseAction.SET_LOWERCASE),
        (case_control_settings.uppercase_mode_key, CaseAction.SET_UPPERCASE),
        (case_control_settings.shift_next_key, CaseAction.TOGGLE_SHIFT_NEXT),
        (case_control_settings.cancel_shift_key, CaseAction.CANCEL_SHIFT),
    )
    for configured_key, action in mappings:
        if is_named_key(key, configured_key):
            return action
    return None


def is_word_confirm_key(key: int, configured_key: str) -> bool:
    normalized = configured_key.strip().lower()
    if normalized == "enter":
        return key in {10, 13}
    return is_named_key(key, normalized)


def is_word_backspace_key(key: int, configured_key: str) -> bool:
    return key in {8, 127} or is_named_key(key, configured_key)


def selected_candidate_rank(key: int) -> int | None:
    candidate_keys = (
        word_builder_settings.candidate_1_key,
        word_builder_settings.candidate_2_key,
        word_builder_settings.candidate_3_key,
    )
    for rank, configured_key in enumerate(candidate_keys, start=1):
        if is_named_key(key, configured_key):
            return rank
    return None


def should_auto_clear_canvas(result: WordBuilderResult) -> bool:
    if result.did_append_character:
        return word_builder_settings.auto_clear_after_commit
    return (
        result.action == WordAction.PENDING_SELECTION_CANCELLED
        and word_builder_settings.auto_clear_after_pending_cancel
    )


def create_air_canvas(width: int, height: int) -> AirCanvas:
    return AirCanvas(
        width=width,
        height=height,
        background_color=settings.canvas_background_color,
        stroke_color=settings.canvas_stroke_color,
        stroke_thickness=settings.canvas_stroke_thickness,
    )


def format_save_status(result: SaveResult) -> str:
    if result.status == SaveStatus.SAVED and result.file_path is not None:
        return f"Saved: {result.file_path.name}"
    return f"Save: {result.status.value}"


def log_save_result(result: SaveResult) -> None:
    if result.status == SaveStatus.SAVED:
        logger.info("%s", result.message)
    elif result.status == SaveStatus.SKIPPED_EMPTY:
        logger.info("%s", result.message)
    elif result.status == SaveStatus.DISABLED:
        logger.debug("%s", result.message)
    else:
        logger.error("%s", result.message)


def make_preprocessed_preview(processed_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cast(
        NDArray[np.uint8],
        cv2.resize(
            processed_image,
            (280, 280),
            interpolation=cv2.INTER_NEAREST,
        ),
    )


def preprocess_canvas_snapshot(
    air_canvas: AirCanvas,
    preprocessor: HandwritingPreprocessor,
    debug_exporter: DebugImageExporter | None,
) -> tuple[str, PreprocessingResult | None]:
    try:
        result = preprocessor.process(air_canvas.get_image(copy=True))
    except EmptyDrawingError:
        logger.info("Nothing to preprocess")
        return "Preprocess: EMPTY", None
    except PreprocessingError as error:
        logger.error("Preprocessing failed: %s", error)
        return "Preprocess: FAILED", None

    cv2.imshow("AirWrite Preprocessed", make_preprocessed_preview(result.processed_image))
    if debug_exporter is not None and result.debug_images:
        exported_paths = debug_exporter.export(result.debug_images)
        logger.info("Exported %s preprocessing debug images", len(exported_paths))
    logger.info(
        "Preprocessed drawing to %sx%s",
        result.processed_image.shape[1],
        result.processed_image.shape[0],
    )
    return "Preprocess: READY", result


def main() -> None:
    settings.create_directories()
    settings.validate_hand_detection_config()
    settings.validate_finger_tracking_config()
    settings.validate_canvas_config()
    settings.validate_gesture_config()
    settings.validate_storage_config()
    settings.validate_preprocessing_config()
    inference_settings.validate()
    identity_model_settings.validate()
    case_control_settings.validate()
    word_builder_settings.validate()
    whole_word_settings.validate()

    logger.info("%s starting", settings.app_name)
    logger.info("Environment: %s", settings.app_env)
    logger.info("Camera index: %s", settings.camera_index)

    camera = CameraStream(
        camera_index=settings.camera_index,
        width=settings.camera_width,
        height=settings.camera_height,
        fps=settings.camera_fps,
        logger=logger,
    )
    processor = FrameProcessor()
    detector: HandDetector | None = None
    renderer = HandLandmarkRenderer(
        draw_landmarks=settings.draw_hand_landmarks,
        draw_connections=settings.draw_hand_connections,
        draw_handedness=False,
    )
    finger_tracker = IndexFingerTracker(
        landmark_index=settings.index_finger_landmark_index,
        smoothing_alpha=settings.finger_smoothing_alpha,
    )
    finger_renderer = FingerTrackingRenderer(
        draw_raw_point=settings.draw_raw_finger_point,
        draw_smoothed_point=settings.draw_smoothed_finger_point,
        point_radius=settings.finger_point_radius,
    )
    gesture_detector = GestureDetector(finger_extension_margin=settings.finger_extension_margin)
    gesture_stabilizer = GestureStabilizer(
        stable_frames=settings.gesture_stable_frames,
        lost_hand_frames=settings.gesture_lost_hand_frames,
    )
    state_machine = DrawingStateMachine(cooldown_ms=settings.gesture_cooldown_ms)
    stroke_manager = StrokeManager(max_point_distance=settings.canvas_max_point_distance)
    overlay_renderer = CanvasOverlayRenderer(
        background_color=settings.canvas_background_color,
        opacity=settings.canvas_overlay_opacity,
    )
    image_saver = DrawingImageSaver(
        output_dir=settings.drawing_output_dir,
        image_format=settings.drawing_image_format,
        filename_prefix=settings.drawing_filename_prefix,
    )
    save_coordinator = DrawingSaveCoordinator(
        saver=image_saver,
        auto_save_on_done=settings.auto_save_on_done,
        enable_manual_save=settings.enable_manual_save,
    )
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
        include_debug_images=settings.save_preprocess_debug_images,
    )
    recognition_service: CharacterRecognitionService | None = None
    predictor: CharacterPredictor | None = None
    case_resolver: CaseLabelResolver | None = None
    case_state = CaseInputState(case_control_settings.default_mode)
    case_controller = CaseModeController(case_state)
    supported_character_set = SupportedCharacterSet.english_letters()
    try:
        custom_bundle = ModelBundleLoader(
            model_path=identity_model_settings.model_path,
            identity_labels_path=identity_model_settings.identity_labels_path,
            lowercase_display_labels_path=identity_model_settings.lowercase_display_labels_path,
            uppercase_display_labels_path=identity_model_settings.uppercase_display_labels_path,
            metadata_path=identity_model_settings.metadata_path,
            preprocessing_config_path=identity_model_settings.preprocessing_config_path,
        ).load()
        ModelBundleValidator().validate(custom_bundle, settings.preprocessing_runtime_contract())
        bundle = custom_bundle
        supported_character_set = SupportedCharacterSet(
            identities=bundle.identity_labels,
            lowercase_characters=bundle.lowercase_display_labels,
            uppercase_characters=bundle.uppercase_display_labels,
        )
        case_resolver = IdentityOnlyLabelResolver(
            bundle.identity_labels,
            bundle.lowercase_display_labels,
            bundle.uppercase_display_labels,
        )
        predictor = CharacterPredictor(bundle=bundle, top_k=inference_settings.top_k)
        emnist_root = identity_model_settings.emnist_model_path.parent
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
        recognition_service = CharacterRecognitionService(
            preprocessor=preprocessor,
            predictor=predictor,
            policy=PredictionPolicy(
                min_confidence=inference_settings.min_confidence,
                min_margin=inference_settings.min_margin,
            ),
            case_resolver=case_resolver,
            logger=logger,
            log_latency=inference_settings.log_latency,
            latency_warning_ms=inference_settings.latency_warning_ms,
        )
        logger.info(
            "Character model v%s loaded once from %s",
            bundle.model_version,
            identity_model_settings.model_path,
        )
    except InferenceError:
        logger.exception("Character prediction is disabled because the model bundle is unavailable")

    completion_coordinator = CompletionRecognitionCoordinator(
        save_coordinator=save_coordinator,
        recognition_service=recognition_service,
        case_state=case_state,
        auto_predict_on_done=inference_settings.auto_predict_on_done,
        manual_predict_enabled=inference_settings.manual_predict_enabled,
    )
    prediction_renderer = PredictionStatusRenderer(
        show_top_k=inference_settings.show_top_k_predictions,
        status_display_ms=inference_settings.status_display_ms,
        show_case_mode=case_control_settings.show_case_mode,
        show_case_debug=case_control_settings.show_case_debug,
    )
    word_builder = WordBuilder(
        supported_character_set=supported_character_set,
        max_length=word_builder_settings.max_length,
    )
    word_builder_controller = WordBuilderController(
        word_builder=word_builder,
        auto_append_accepted=word_builder_settings.auto_append_accepted,
        require_selection_for_uncertain=(word_builder_settings.require_selection_for_uncertain),
    )
    input_mode_controller = InputModeController(
        DrawingInputMode.from_string(whole_word_settings.default_input_mode)
    )
    word_case_policy = WordCasePolicy.from_string(whole_word_settings.default_case_policy)
    stroke_recorder = StrokeRecorder(
        max_strokes=whole_word_settings.max_strokes,
        max_points_per_stroke=whole_word_settings.max_points_per_stroke,
    )
    whole_word_controller: WholeWordController | None = None
    if predictor is not None and case_resolver is not None:
        word_strategy = IsolatedLetterWordRecognitionStrategy(
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
            batch_builder=CharacterBatchBuilder(
                preprocessor,
                max_characters=whole_word_settings.max_characters,
            ),
            predictor=predictor,
            prediction_policy=PredictionPolicy(
                min_confidence=whole_word_settings.min_confidence,
                min_margin=whole_word_settings.min_margin,
            ),
            label_resolver=case_resolver,
        )
        whole_word_controller = WholeWordController(
            recognition_service=WholeWordRecognitionService(word_strategy),
            correction_strategy=word_strategy,
            word_builder=word_builder,
            min_characters=whole_word_settings.min_characters,
            max_characters=whole_word_settings.max_characters,
        )
    whole_word_renderer = WholeWordRenderer(
        show_roi=whole_word_settings.show_roi,
        show_segment_boxes=whole_word_settings.show_segment_boxes,
        show_prediction_confidence=whole_word_settings.show_prediction_confidence,
    )
    status_hud_renderer = StatusHudRenderer(target_font_height_px=13)
    debug_exporter = (
        DebugImageExporter(settings.preprocess_debug_output_dir)
        if settings.save_preprocess_debug_images
        else None
    )
    dataset_saver = (
        CharacterDatasetImageSaver(
            output_dir=settings.dataset_capture_output_dir,
            image_format=settings.dataset_capture_image_format,
            filename_prefix=settings.dataset_capture_filename_prefix,
            writing_style=settings.dataset_capture_writing_style,
        )
        if settings.enable_dataset_capture
        else None
    )
    air_canvas: AirCanvas | None = None
    drawing_controller: DrawingController | None = None
    last_save_result: SaveResult | None = None
    last_preprocess_status: str | None = None
    last_dataset_status: str | None = None
    last_prediction_result: PredictionResult | None = None
    last_word_builder_result = word_builder.snapshot()
    selected_dataset_label = (
        dataset_saver.normalize_capture_label(settings.dataset_capture_initial_label)
        if dataset_saver is not None
        else settings.dataset_capture_initial_label
    )
    dataset_samples_saved = 0
    save_status_expires_at_ms = 0
    preprocess_status_expires_at_ms = 0
    dataset_status_expires_at_ms = 0
    prediction_display_started_ms = 0
    case_status_expires_at_ms = 0
    last_case_status: str | None = None
    word_status_started_ms = 0
    previous_timestamp_ms = 0
    word_writing_region: WordWritingRegion | None = None
    word_done_handled = False

    try:
        detector = HandDetector(
            model_path=settings.hand_landmarker_model_path,
            num_hands=settings.hand_num_hands,
            min_detection_confidence=settings.hand_min_detection_confidence,
            min_presence_confidence=settings.hand_min_presence_confidence,
            min_tracking_confidence=settings.hand_min_tracking_confidence,
            logger=logger,
        )
        camera.open()
        cv2.namedWindow(settings.camera_window_name, cv2.WINDOW_NORMAL)

        while True:
            success, frame = camera.read()
            if not success or not processor.validate_frame(frame):
                logger.error("Stopping camera loop because frame is invalid")
                break

            assert frame is not None
            mirrored_frame = processor.mirror_frame(frame, settings.camera_mirror)
            frame_height, frame_width = mirrored_frame.shape[:2]
            if air_canvas is None:
                air_canvas = create_air_canvas(width=frame_width, height=frame_height)
                drawing_controller = DrawingController(state_machine, stroke_manager, air_canvas)
                word_writing_region = WordWritingRegion.from_ratios(
                    frame_width,
                    frame_height,
                    x_ratio=whole_word_settings.roi_x_ratio,
                    y_ratio=whole_word_settings.roi_y_ratio,
                    width_ratio=whole_word_settings.roi_width_ratio,
                    height_ratio=whole_word_settings.roi_height_ratio,
                )
            elif not air_canvas.matches_size(width=frame_width, height=frame_height):
                air_canvas.reset_size(width=frame_width, height=frame_height)
                stroke_manager.reset()
                stroke_recorder.reset()
                word_writing_region = WordWritingRegion.from_ratios(
                    frame_width,
                    frame_height,
                    x_ratio=whole_word_settings.roi_x_ratio,
                    y_ratio=whole_word_settings.roi_y_ratio,
                    width_ratio=whole_word_settings.roi_width_ratio,
                    height_ratio=whole_word_settings.roi_height_ratio,
                )
                if drawing_controller is not None:
                    drawing_controller.set_state(DrawingState.READY, previous_timestamp_ms)

            previous_timestamp_ms = next_timestamp_ms(previous_timestamp_ms)
            hand_result = detector.detect(mirrored_frame, previous_timestamp_ms)
            raw_gesture = gesture_detector.detect(hand_result)
            stable_gesture = (
                gesture_stabilizer.update(raw_gesture)
                if settings.enable_gesture_control
                else Gesture.INDEX_ONLY
                if hand_result.has_hands
                else Gesture.NO_HAND
            )
            finger_result = finger_tracker.track(
                hand_result,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            assert drawing_controller is not None
            word_mode = input_mode_controller.current_mode is DrawingInputMode.ISOLATED_WORD
            drawing_controller.configure_word_input(
                stroke_recorder if word_mode else None,
                word_writing_region if word_mode else None,
            )
            if whole_word_controller is not None and whole_word_controller.draft is not None:
                controller_result = drawing_controller.set_state(
                    DrawingState.PAUSED,
                    previous_timestamp_ms,
                )
            else:
                controller_result = drawing_controller.update(stable_gesture, finger_result)
            if controller_result.state is not DrawingState.DONE:
                word_done_handled = False
            completion_result = (
                completion_coordinator.handle_transition(
                    controller_result.state,
                    air_canvas,
                    allow_prediction=word_builder.pending_selection is None,
                )
                if not word_mode
                else None
            )
            if (
                word_mode
                and controller_result.state is DrawingState.DONE
                and not word_done_handled
                and whole_word_controller is not None
                and word_writing_region is not None
            ):
                stroke_recorder.end_stroke()
                whole_word_controller.recognize(
                    WordInputSnapshot(
                        snapshot_id=uuid4().hex,
                        canvas_image=air_canvas.get_image(copy=True),
                        strokes=stroke_recorder.snapshot(),
                        writing_region=word_writing_region,
                        completed_at_ms=previous_timestamp_ms,
                    ),
                    word_case_policy,
                )
                word_done_handled = True
                logger.info("%s", whole_word_controller.last_message)
            if completion_result is not None and completion_result.save_result is not None:
                save_result = completion_result.save_result
                last_save_result = save_result
                save_status_expires_at_ms = previous_timestamp_ms + settings.save_status_display_ms
                log_save_result(save_result)
            if completion_result is not None and completion_result.prediction_result is not None:
                last_prediction_result = completion_result.prediction_result
                prediction_display_started_ms = previous_timestamp_ms
                logger.info("%s", last_prediction_result.message)
                last_word_builder_result = word_builder_controller.handle_prediction(
                    last_prediction_result
                )
                word_status_started_ms = previous_timestamp_ms
                logger.info("%s", last_word_builder_result.message)
                update_case_state_after_word_action(last_word_builder_result, case_state)
                if should_auto_clear_canvas(last_word_builder_result):
                    controller_result = drawing_controller.clear(previous_timestamp_ms)

            annotated_frame = renderer.draw(mirrored_frame, hand_result)
            annotated_frame = finger_renderer.draw(annotated_frame, finger_result)
            canvas_image = air_canvas.get_image()
            if settings.show_camera_with_canvas:
                annotated_frame = overlay_renderer.render(annotated_frame, canvas_image)

            fps = processor.calculate_fps()
            hand_label = (
                hand_result.hands[0].handedness
                if hand_result.has_hands and hand_result.hands[0].handedness
                else "NONE"
            )
            finger_status = "TRACKED" if finger_result.is_tracked else "NOT DETECTED"
            save_status_lines = (
                [format_save_status(last_save_result)]
                if settings.show_save_status
                and last_save_result is not None
                and previous_timestamp_ms <= save_status_expires_at_ms
                else []
            )
            preprocess_status_lines = (
                [last_preprocess_status]
                if settings.show_save_status
                and last_preprocess_status is not None
                and previous_timestamp_ms <= preprocess_status_expires_at_ms
                else []
            )
            dataset_status_lines = (
                [last_dataset_status]
                if settings.show_save_status
                and last_dataset_status is not None
                and previous_timestamp_ms <= dataset_status_expires_at_ms
                else []
            )
            display_frame = whole_word_renderer.render(
                annotated_frame,
                input_mode_controller.current_mode,
                word_case_policy,
                word_writing_region,
                whole_word_controller.draft if whole_word_controller is not None else None,
                show_status=False,
            )
            active_draft = (
                whole_word_controller.draft if whole_word_controller is not None else None
            )
            hud_lines = [
                *([f"FPS: {fps:.1f}"] if settings.show_fps else []),
                f"Input: {input_mode_controller.current_mode.value}",
                "Switch: W WORD | R CHARACTER",
                (
                    f"Case: {word_case_policy.value}"
                    if input_mode_controller.current_mode is DrawingInputMode.ISOLATED_WORD
                    else f"Case: {case_state.get_effective_mode().value}"
                ),
                f"Word: {word_builder.current_word or '_'}",
                *([f"Hand: {hand_label}"] if settings.draw_handedness else []),
                f"Tracking: {finger_status}",
                f"Canvas: {'EMPTY' if air_canvas.is_empty() else 'DRAWING'}",
                f"State: {controller_result.state.value}",
                *save_status_lines,
                *preprocess_status_lines,
                *dataset_status_lines,
            ]
            if active_draft is not None:
                selected = active_draft.selected_character
                hud_lines.extend(
                    [
                        f"Draft: {active_draft.current_word}",
                        (
                            "Position: "
                            f"{active_draft.selected_position + 1}/"
                            f"{len(active_draft.characters)}"
                        ),
                        "Candidates: "
                        + "  ".join(
                            f"{candidate.rank}:{candidate.rendered_character}"
                            for candidate in selected.candidates
                        ),
                        f"Confidence: {selected.confidence:.1%}",
                        f"Review: {'READY' if active_draft.can_accept else 'REQUIRED'}",
                    ]
                )
            elif word_builder.pending_selection is not None:
                hud_lines.append(
                    "Candidates: "
                    + "  ".join(
                        f"{candidate.rank}:{candidate.rendered_character}"
                        for candidate in word_builder.pending_selection.candidates
                    )
                )
            elif (
                prediction_renderer.is_visible(
                    last_prediction_result,
                    prediction_display_started_ms,
                    previous_timestamp_ms,
                )
                and last_prediction_result is not None
            ):
                hud_lines.extend(prediction_renderer.format_lines(last_prediction_result))
            if last_case_status is not None and previous_timestamp_ms <= case_status_expires_at_ms:
                hud_lines.append(last_case_status)
            if (
                previous_timestamp_ms >= word_status_started_ms
                and previous_timestamp_ms - word_status_started_ms
                <= word_builder_settings.status_display_ms
                and last_word_builder_result.message != "Word builder ready"
            ):
                hud_lines.append(last_word_builder_result.message)
            display_frame = status_hud_renderer.render(display_frame, hud_lines)

            cv2.imshow(settings.camera_window_name, display_frame)
            if settings.show_canvas_window:
                cv2.imshow("AirWrite Canvas", canvas_image)

            key = cv2.waitKey(1) & 0xFF
            if should_quit(settings.camera_window_name, key):
                logger.info("Camera loop stopped by user")
                break

            if active_draft is not None:
                rank = selected_candidate_rank(key)
                try:
                    if is_named_key(key, whole_word_settings.previous_position_key):
                        whole_word_controller.move_previous()
                    elif is_named_key(key, whole_word_settings.next_position_key):
                        whole_word_controller.move_next()
                    elif rank is not None:
                        whole_word_controller.select_candidate(rank)
                    elif is_named_key(key, whole_word_settings.toggle_case_key):
                        whole_word_controller.toggle_case()
                    elif is_named_key(key, whole_word_settings.split_segment_key):
                        whole_word_controller.split_selected()
                    elif is_named_key(key, whole_word_settings.merge_segment_key):
                        whole_word_controller.merge_selected_with_next()
                    elif is_named_key(key, whole_word_settings.accept_draft_key):
                        last_word_builder_result = whole_word_controller.accept()
                        word_status_started_ms = previous_timestamp_ms
                        if last_word_builder_result.did_commit_word:
                            drawing_controller.clear(previous_timestamp_ms)
                            word_done_handled = False
                    elif is_named_key(key, whole_word_settings.cancel_draft_key):
                        whole_word_controller.cancel()
                        drawing_controller.clear(previous_timestamp_ms)
                        word_done_handled = False
                    else:
                        continue
                except (RuntimeError, ValueError) as error:
                    whole_word_controller.last_message = str(error)
                    logger.warning("Whole-word review action failed: %s", error)
                continue

            if is_named_key(key, whole_word_settings.character_mode_key):
                if word_builder.pending_selection is not None:
                    cancelled_pending = word_builder.pending_selection
                    last_word_builder_result = word_builder_controller.cancel_pending()
                    update_case_state_after_word_action(
                        last_word_builder_result,
                        case_state,
                        cancelled_pending,
                    )
                if input_mode_controller.set_character_mode():
                    drawing_controller.clear(previous_timestamp_ms)
                    stroke_recorder.reset()
                    word_done_handled = False
                    last_word_builder_result = word_builder.snapshot(
                        message="Character Mode enabled"
                    )
                    word_status_started_ms = previous_timestamp_ms
                    logger.info("Drawing input mode changed to CHARACTER")
                continue
            if is_named_key(key, whole_word_settings.word_mode_key):
                if word_builder.pending_selection is not None:
                    cancelled_pending = word_builder.pending_selection
                    last_word_builder_result = word_builder_controller.cancel_pending()
                    update_case_state_after_word_action(
                        last_word_builder_result,
                        case_state,
                        cancelled_pending,
                    )
                if input_mode_controller.set_word_mode():
                    drawing_controller.clear(previous_timestamp_ms)
                    stroke_recorder.reset()
                    word_done_handled = False
                    last_word_builder_result = word_builder.snapshot(
                        message="Whole-Word Mode enabled; write inside the ROI"
                    )
                    word_status_started_ms = previous_timestamp_ms
                    logger.info("Drawing input mode changed to ISOLATED_WORD")
                continue

            if input_mode_controller.current_mode is DrawingInputMode.ISOLATED_WORD:
                word_case_keys = {
                    whole_word_settings.case_lowercase_key: WordCasePolicy.LOWERCASE,
                    whole_word_settings.case_uppercase_key: WordCasePolicy.UPPERCASE,
                    whole_word_settings.case_capitalize_key: WordCasePolicy.CAPITALIZE_FIRST,
                    whole_word_settings.case_custom_key: WordCasePolicy.CUSTOM,
                }
                selected_policy = next(
                    (
                        policy
                        for configured_key, policy in word_case_keys.items()
                        if is_named_key(key, configured_key)
                    ),
                    None,
                )
                if selected_policy is not None:
                    word_case_policy = selected_policy
                    logger.info("Whole-word case policy changed to %s", word_case_policy.value)
                    continue

            if word_builder.pending_selection is not None:
                rank = selected_candidate_rank(key)
                if rank is not None:
                    last_word_builder_result = word_builder_controller.select_candidate(rank)
                    word_status_started_ms = previous_timestamp_ms
                    logger.info("%s", last_word_builder_result.message)
                    update_case_state_after_word_action(last_word_builder_result, case_state)
                    if should_auto_clear_canvas(last_word_builder_result):
                        drawing_controller.clear(previous_timestamp_ms)
                    continue
                if is_named_key(key, word_builder_settings.cancel_pending_key):
                    cancelled_pending = word_builder.pending_selection
                    last_word_builder_result = word_builder_controller.cancel_pending()
                    word_status_started_ms = previous_timestamp_ms
                    logger.info("%s", last_word_builder_result.message)
                    update_case_state_after_word_action(
                        last_word_builder_result,
                        case_state,
                        cancelled_pending,
                    )
                    if should_auto_clear_canvas(last_word_builder_result):
                        drawing_controller.clear(previous_timestamp_ms)
                    continue

            case_action = (
                case_action_for_key(key)
                if input_mode_controller.current_mode is DrawingInputMode.CHARACTER
                else None
            )
            if case_action is not None:
                case_result = case_controller.handle(case_action)
                last_case_status = case_result.message
                case_status_expires_at_ms = (
                    previous_timestamp_ms + case_control_settings.status_display_ms
                )
                logger.info("%s", case_result.message)
                continue

            if is_word_backspace_key(key, word_builder_settings.backspace_key):
                last_word_builder_result = word_builder_controller.backspace()
                word_status_started_ms = previous_timestamp_ms
                logger.info("%s", last_word_builder_result.message)
                continue
            if is_named_key(key, word_builder_settings.clear_word_key):
                last_word_builder_result = word_builder_controller.clear_word()
                update_case_state_after_word_action(last_word_builder_result, case_state)
                word_status_started_ms = previous_timestamp_ms
                logger.info("%s", last_word_builder_result.message)
                continue
            if is_word_confirm_key(key, word_builder_settings.confirm_key):
                last_word_builder_result = word_builder_controller.confirm(previous_timestamp_ms)
                word_status_started_ms = previous_timestamp_ms
                logger.info("%s", last_word_builder_result.message)
                continue
            if is_named_key(key, word_builder_settings.new_word_key):
                last_word_builder_result = word_builder_controller.start_new_word()
                update_case_state_after_word_action(last_word_builder_result, case_state)
                word_status_started_ms = previous_timestamp_ms
                logger.info("%s", last_word_builder_result.message)
                continue

            if (
                settings.enable_keyboard_fallback
                and settings.enable_manual_save
                and drawing_controller is not None
                and is_manual_save_key(key, settings.manual_save_key)
            ):
                save_result = save_coordinator.save_now(air_canvas)
                last_save_result = save_result
                save_status_expires_at_ms = previous_timestamp_ms + settings.save_status_display_ms
                log_save_result(save_result)
                if save_result.success:
                    last_preprocess_status, _preprocess_result = preprocess_canvas_snapshot(
                        air_canvas,
                        preprocessor,
                        debug_exporter,
                    )
                    preprocess_status_expires_at_ms = (
                        previous_timestamp_ms + settings.save_status_display_ms
                    )
                    if settings.clear_canvas_after_save:
                        drawing_controller.clear(previous_timestamp_ms)
                continue
            if (
                settings.enable_keyboard_fallback
                and inference_settings.manual_predict_enabled
                and input_mode_controller.current_mode is DrawingInputMode.CHARACTER
                and drawing_controller is not None
                and is_manual_predict_key(key, inference_settings.manual_predict_key)
            ):
                if word_builder.pending_selection is not None:
                    last_word_builder_result = word_builder.snapshot(
                        message="Resolve or cancel the pending character before predicting again"
                    )
                    word_status_started_ms = previous_timestamp_ms
                    logger.info("%s", last_word_builder_result.message)
                    continue
                prediction_result = completion_coordinator.predict_now(air_canvas)
                if prediction_result is not None:
                    last_prediction_result = prediction_result
                    prediction_display_started_ms = previous_timestamp_ms
                    logger.info("Manual prediction: %s", prediction_result.message)
                    last_word_builder_result = word_builder_controller.handle_prediction(
                        prediction_result
                    )
                    word_status_started_ms = previous_timestamp_ms
                    logger.info("%s", last_word_builder_result.message)
                    update_case_state_after_word_action(last_word_builder_result, case_state)
                    if should_auto_clear_canvas(last_word_builder_result):
                        drawing_controller.clear(previous_timestamp_ms)
                continue
            if (
                settings.enable_keyboard_fallback
                and drawing_controller is not None
                and is_manual_preprocess_key(key, settings.manual_preprocess_key)
            ):
                last_preprocess_status, _preprocess_result = preprocess_canvas_snapshot(
                    air_canvas,
                    preprocessor,
                    debug_exporter,
                )
                preprocess_status_expires_at_ms = (
                    previous_timestamp_ms + settings.save_status_display_ms
                )
                continue
            if (
                settings.enable_keyboard_fallback
                and settings.enable_dataset_capture
                and settings.dataset_capture_limit == 0
                and dataset_saver is not None
                and is_dataset_label_previous_key(key, settings.dataset_capture_previous_label_key)
            ):
                selected_dataset_label = dataset_saver.previous_capture_label(
                    selected_dataset_label
                )
                last_dataset_status = f"Dataset Label: {selected_dataset_label}"
                dataset_status_expires_at_ms = (
                    previous_timestamp_ms + settings.save_status_display_ms
                )
                logger.info("Dataset capture label changed to %s", selected_dataset_label)
                continue
            if (
                settings.enable_keyboard_fallback
                and settings.enable_dataset_capture
                and settings.dataset_capture_limit == 0
                and dataset_saver is not None
                and is_dataset_label_next_key(key, settings.dataset_capture_next_label_key)
            ):
                selected_dataset_label = dataset_saver.next_capture_label(selected_dataset_label)
                last_dataset_status = f"Dataset Label: {selected_dataset_label}"
                dataset_status_expires_at_ms = (
                    previous_timestamp_ms + settings.save_status_display_ms
                )
                logger.info("Dataset capture label changed to %s", selected_dataset_label)
                continue
            if (
                settings.enable_keyboard_fallback
                and settings.enable_dataset_capture
                and dataset_saver is not None
                and drawing_controller is not None
                and is_dataset_capture_key(key, settings.dataset_capture_save_key)
            ):
                last_preprocess_status, preprocess_result = preprocess_canvas_snapshot(
                    air_canvas,
                    preprocessor,
                    debug_exporter,
                )
                preprocess_status_expires_at_ms = (
                    previous_timestamp_ms + settings.save_status_display_ms
                )
                if preprocess_result is None:
                    last_dataset_status = "Dataset: SKIPPED"
                else:
                    try:
                        dataset_result = dataset_saver.save(
                            selected_dataset_label,
                            preprocess_result.processed_image,
                        )
                    except (DrawingStorageError, ValueError) as error:
                        last_dataset_status = "Dataset: FAILED"
                        logger.error("Dataset capture failed: %s", error)
                    else:
                        last_dataset_status = (
                            f"Dataset: {dataset_result.label}/{dataset_result.file_path.name}"
                        )
                        logger.info("%s", dataset_result.message)
                        dataset_samples_saved += 1
                        if settings.dataset_capture_clear_after_save:
                            drawing_controller.clear(previous_timestamp_ms)
                        if (
                            settings.dataset_capture_limit > 0
                            and dataset_samples_saved >= settings.dataset_capture_limit
                        ):
                            logger.info(
                                "Dataset collection complete: %s/%s samples for %s",
                                dataset_samples_saved,
                                settings.dataset_capture_limit,
                                selected_dataset_label,
                            )
                            break
                dataset_status_expires_at_ms = (
                    previous_timestamp_ms + settings.save_status_display_ms
                )
                continue
            if (
                drawing_controller is not None
                and is_clear_key(key, settings.canvas_clear_key)
                and settings.enable_keyboard_fallback
            ):
                drawing_controller.clear(previous_timestamp_ms)
                logger.info("Canvas cleared")
                continue
            if (
                settings.enable_keyboard_fallback
                and drawing_controller is not None
                and is_space_key(key)
            ):
                next_state = (
                    DrawingState.PAUSED
                    if state_machine.state == DrawingState.WRITING
                    else DrawingState.WRITING
                )
                drawing_controller.set_state(next_state, previous_timestamp_ms)
                logger.info("Keyboard fallback changed state to %s", next_state.value)
                continue
            if (
                settings.enable_keyboard_fallback
                and drawing_controller is not None
                and is_done_key(key)
            ):
                controller_result = drawing_controller.set_state(
                    DrawingState.DONE,
                    previous_timestamp_ms,
                )
                if input_mode_controller.current_mode is DrawingInputMode.ISOLATED_WORD:
                    if whole_word_controller is None or word_writing_region is None:
                        logger.error("Whole-word prediction is unavailable")
                    else:
                        stroke_recorder.end_stroke()
                        whole_word_controller.recognize(
                            WordInputSnapshot(
                                snapshot_id=uuid4().hex,
                                canvas_image=air_canvas.get_image(copy=True),
                                strokes=stroke_recorder.snapshot(),
                                writing_region=word_writing_region,
                                completed_at_ms=previous_timestamp_ms,
                            ),
                            word_case_policy,
                        )
                        word_done_handled = True
                        logger.info("%s", whole_word_controller.last_message)
                    continue
                completion_result = completion_coordinator.handle_transition(
                    controller_result.state,
                    air_canvas,
                    allow_prediction=word_builder.pending_selection is None,
                )
                if completion_result is not None and completion_result.save_result is not None:
                    save_result = completion_result.save_result
                    last_save_result = save_result
                    save_status_expires_at_ms = (
                        previous_timestamp_ms + settings.save_status_display_ms
                    )
                    log_save_result(save_result)
                if (
                    completion_result is not None
                    and completion_result.prediction_result is not None
                ):
                    last_prediction_result = completion_result.prediction_result
                    prediction_display_started_ms = previous_timestamp_ms
                    logger.info("%s", last_prediction_result.message)
                    last_word_builder_result = word_builder_controller.handle_prediction(
                        last_prediction_result
                    )
                    word_status_started_ms = previous_timestamp_ms
                    logger.info("%s", last_word_builder_result.message)
                    update_case_state_after_word_action(last_word_builder_result, case_state)
                    if should_auto_clear_canvas(last_word_builder_result):
                        drawing_controller.clear(previous_timestamp_ms)
                logger.info("Keyboard fallback changed state to DONE")
                continue
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
    finally:
        if detector is not None:
            detector.close()
        gesture_stabilizer.reset()
        finger_tracker.reset()
        stroke_manager.reset()
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Camera and hand detector resources released")


if __name__ == "__main__":
    main()
