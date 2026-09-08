import os
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

from app.inference.character_case_mode import CharacterCaseMode

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def env_to_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False

    raise ValueError(
        f"Invalid boolean value for {name}: {value!r}. Use true/false, 1/0, yes/no, or on/off."
    )


def env_to_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_to_color(name: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    value = os.getenv(name)
    if value is None:
        return default

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain three comma-separated values")

    color = cast(tuple[int, int, int], tuple(int(part) for part in parts))
    validate_color(name, color)
    return color


def validate_color(name: str, color: tuple[int, int, int]) -> None:
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError(f"{name} channels must be between 0 and 255, got {color}")


def validate_confidence(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")


@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "AirWrite Vocabulary Assistant")
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    camera_width: int = int(os.getenv("CAMERA_WIDTH", "1280"))
    camera_height: int = int(os.getenv("CAMERA_HEIGHT", "720"))
    camera_fps: int = int(os.getenv("CAMERA_FPS", "30"))
    camera_mirror: bool = env_to_bool("CAMERA_MIRROR", True)
    camera_window_name: str = os.getenv("CAMERA_WINDOW_NAME", "AirWrite Camera")
    show_fps: bool = env_to_bool("SHOW_FPS", True)

    hand_landmarker_model_path: Path = PROJECT_ROOT / os.getenv(
        "HAND_LANDMARKER_MODEL_PATH", "models/hand_landmarker.task"
    )
    hand_num_hands: int = int(os.getenv("HAND_NUM_HANDS", "1"))
    hand_min_detection_confidence: float = env_to_float("HAND_MIN_DETECTION_CONFIDENCE", 0.5)
    hand_min_presence_confidence: float = env_to_float("HAND_MIN_PRESENCE_CONFIDENCE", 0.5)
    hand_min_tracking_confidence: float = env_to_float("HAND_MIN_TRACKING_CONFIDENCE", 0.5)
    draw_hand_landmarks: bool = env_to_bool("DRAW_HAND_LANDMARKS", True)
    draw_hand_connections: bool = env_to_bool("DRAW_HAND_CONNECTIONS", True)
    draw_handedness: bool = env_to_bool("DRAW_HANDEDNESS", True)
    index_finger_landmark_index: int = int(os.getenv("INDEX_FINGER_LANDMARK_INDEX", "8"))
    finger_smoothing_alpha: float = env_to_float("FINGER_SMOOTHING_ALPHA", 0.5)
    draw_raw_finger_point: bool = env_to_bool("DRAW_RAW_FINGER_POINT", False)
    draw_smoothed_finger_point: bool = env_to_bool("DRAW_SMOOTHED_FINGER_POINT", True)
    finger_point_radius: int = int(os.getenv("FINGER_POINT_RADIUS", "8"))
    canvas_background_color: tuple[int, int, int] = env_to_color(
        "CANVAS_BACKGROUND_COLOR", (0, 0, 0)
    )
    canvas_stroke_color: tuple[int, int, int] = env_to_color("CANVAS_STROKE_COLOR", (255, 255, 255))
    canvas_stroke_thickness: int = int(os.getenv("CANVAS_STROKE_THICKNESS", "8"))
    canvas_max_point_distance: float = env_to_float("CANVAS_MAX_POINT_DISTANCE", 120.0)
    canvas_overlay_opacity: float = env_to_float("CANVAS_OVERLAY_OPACITY", 1.0)
    show_camera_with_canvas: bool = env_to_bool("SHOW_CAMERA_WITH_CANVAS", True)
    show_canvas_window: bool = env_to_bool("SHOW_CANVAS_WINDOW", True)
    canvas_clear_key: str = os.getenv("CANVAS_CLEAR_KEY", "c")
    gesture_stable_frames: int = int(os.getenv("GESTURE_STABLE_FRAMES", "5"))
    gesture_cooldown_ms: int = int(os.getenv("GESTURE_COOLDOWN_MS", "500"))
    gesture_lost_hand_frames: int = int(os.getenv("GESTURE_LOST_HAND_FRAMES", "10"))
    enable_gesture_control: bool = env_to_bool("ENABLE_GESTURE_CONTROL", True)
    enable_keyboard_fallback: bool = env_to_bool("ENABLE_KEYBOARD_FALLBACK", True)
    draw_gesture_label: bool = env_to_bool("DRAW_GESTURE_LABEL", True)
    draw_state_label: bool = env_to_bool("DRAW_STATE_LABEL", True)
    clear_hold_ms: int = int(os.getenv("CLEAR_HOLD_MS", "1500"))
    enable_clear_gesture: bool = env_to_bool("ENABLE_CLEAR_GESTURE", False)
    finger_extension_margin: float = env_to_float("FINGER_EXTENSION_MARGIN", 0.02)
    drawing_output_dir: Path = PROJECT_ROOT / os.getenv("DRAWING_OUTPUT_DIR", "data/drawings")
    drawing_image_format: str = os.getenv("DRAWING_IMAGE_FORMAT", "png")
    drawing_filename_prefix: str = os.getenv("DRAWING_FILENAME_PREFIX", "drawing")
    auto_save_on_done: bool = env_to_bool("AUTO_SAVE_ON_DONE", True)
    enable_manual_save: bool = env_to_bool("ENABLE_MANUAL_SAVE", True)
    manual_save_key: str = os.getenv("MANUAL_SAVE_KEY", "s")
    clear_canvas_after_save: bool = env_to_bool("CLEAR_CANVAS_AFTER_SAVE", False)
    show_save_status: bool = env_to_bool("SHOW_SAVE_STATUS", True)
    save_status_display_ms: int = int(os.getenv("SAVE_STATUS_DISPLAY_MS", "2000"))
    preprocess_output_width: int = int(
        os.getenv("MODEL_INPUT_WIDTH", os.getenv("PREPROCESS_OUTPUT_WIDTH", "28"))
    )
    preprocess_output_height: int = int(
        os.getenv("MODEL_INPUT_HEIGHT", os.getenv("PREPROCESS_OUTPUT_HEIGHT", "28"))
    )
    preprocess_content_width: int = int(
        os.getenv("AIRWRITE_CONTENT_WIDTH", os.getenv("PREPROCESS_CONTENT_WIDTH", "20"))
    )
    preprocess_content_height: int = int(
        os.getenv("AIRWRITE_CONTENT_HEIGHT", os.getenv("PREPROCESS_CONTENT_HEIGHT", "20"))
    )
    preprocess_binary_threshold: int = int(
        os.getenv("AIRWRITE_BINARY_THRESHOLD", os.getenv("PREPROCESS_BINARY_THRESHOLD", "20"))
    )
    preprocess_crop_padding: int = int(
        os.getenv("AIRWRITE_CROP_PADDING", os.getenv("PREPROCESS_CROP_PADDING", "8"))
    )
    preprocess_min_foreground_pixels: int = int(
        os.getenv(
            "AIRWRITE_MIN_FOREGROUND_PIXELS",
            os.getenv("PREPROCESS_MIN_FOREGROUND_PIXELS", "10"),
        )
    )
    preprocess_invert_input: bool = env_to_bool(
        "AIRWRITE_INVERT_INPUT", env_to_bool("PREPROCESS_INVERT_INPUT", False)
    )
    preprocess_center_of_mass: bool = env_to_bool(
        "AIRWRITE_CENTER_OF_MASS", env_to_bool("PREPROCESS_CENTER_OF_MASS", False)
    )
    save_preprocess_debug_images: bool = env_to_bool("SAVE_PREPROCESS_DEBUG_IMAGES", False)
    preprocess_debug_output_dir: Path = PROJECT_ROOT / os.getenv(
        "PREPROCESS_DEBUG_OUTPUT_DIR", "data/preprocessed_debug"
    )
    manual_preprocess_key: str = os.getenv("MANUAL_PREPROCESS_KEY", "p")
    enable_dataset_capture: bool = env_to_bool("ENABLE_DATASET_CAPTURE", True)
    dataset_capture_output_dir: Path = PROJECT_ROOT / os.getenv(
        "DATASET_CAPTURE_OUTPUT_DIR", "data/raw_airwrite"
    )
    dataset_capture_image_format: str = os.getenv("DATASET_CAPTURE_IMAGE_FORMAT", "png")
    dataset_capture_filename_prefix: str = os.getenv("DATASET_CAPTURE_FILENAME_PREFIX", "airwrite")
    dataset_capture_writing_style: str = os.getenv("DATASET_CAPTURE_WRITING_STYLE", "uppercase")
    dataset_capture_initial_label: str = os.getenv("DATASET_CAPTURE_INITIAL_LABEL", "A")
    dataset_capture_save_key: str = os.getenv("DATASET_CAPTURE_SAVE_KEY", "v")
    dataset_capture_next_label_key: str = os.getenv("DATASET_CAPTURE_NEXT_LABEL_KEY", "]")
    dataset_capture_previous_label_key: str = os.getenv("DATASET_CAPTURE_PREVIOUS_LABEL_KEY", "[")
    dataset_capture_clear_after_save: bool = env_to_bool("DATASET_CAPTURE_CLEAR_AFTER_SAVE", False)
    dataset_capture_limit: int = int(os.getenv("DATASET_CAPTURE_LIMIT", "0"))

    model_path: Path = PROJECT_ROOT / os.getenv("MODEL_PATH", "models/character_cnn.pth")
    raw_data_dir: Path = PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
    processed_data_dir: Path = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
    saved_drawings_dir: Path = PROJECT_ROOT / os.getenv("SAVED_DRAWINGS_DIR", "data/saved_drawings")

    def create_directories(self) -> None:
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.saved_drawings_dir.mkdir(parents=True, exist_ok=True)
        if self.save_preprocess_debug_images:
            self.preprocess_debug_output_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_dataset_capture:
            self.dataset_capture_output_dir.mkdir(parents=True, exist_ok=True)

    def validate_hand_detection_config(self) -> None:
        if self.hand_num_hands <= 0:
            raise ValueError(f"HAND_NUM_HANDS must be greater than 0, got {self.hand_num_hands}")

        validate_confidence("HAND_MIN_DETECTION_CONFIDENCE", self.hand_min_detection_confidence)
        validate_confidence("HAND_MIN_PRESENCE_CONFIDENCE", self.hand_min_presence_confidence)
        validate_confidence("HAND_MIN_TRACKING_CONFIDENCE", self.hand_min_tracking_confidence)

    def validate_finger_tracking_config(self) -> None:
        if not 0 <= self.index_finger_landmark_index <= 20:
            raise ValueError(
                "INDEX_FINGER_LANDMARK_INDEX must be between 0 and 20, "
                f"got {self.index_finger_landmark_index}"
            )
        if not 0.0 < self.finger_smoothing_alpha <= 1.0:
            raise ValueError(
                "FINGER_SMOOTHING_ALPHA must be greater than 0.0 and less than or equal "
                f"to 1.0, got {self.finger_smoothing_alpha}"
            )
        if self.finger_point_radius <= 0:
            raise ValueError(
                f"FINGER_POINT_RADIUS must be greater than 0, got {self.finger_point_radius}"
            )

    def validate_canvas_config(self) -> None:
        validate_color("CANVAS_BACKGROUND_COLOR", self.canvas_background_color)
        validate_color("CANVAS_STROKE_COLOR", self.canvas_stroke_color)
        if self.canvas_stroke_thickness <= 0:
            raise ValueError(
                "CANVAS_STROKE_THICKNESS must be greater than 0, "
                f"got {self.canvas_stroke_thickness}"
            )
        if self.canvas_max_point_distance <= 0:
            raise ValueError(
                "CANVAS_MAX_POINT_DISTANCE must be greater than 0, "
                f"got {self.canvas_max_point_distance}"
            )
        if not 0.0 <= self.canvas_overlay_opacity <= 1.0:
            raise ValueError(
                "CANVAS_OVERLAY_OPACITY must be between 0.0 and 1.0, "
                f"got {self.canvas_overlay_opacity}"
            )
        if len(self.canvas_clear_key) != 1:
            raise ValueError("CANVAS_CLEAR_KEY must contain exactly one character")

    def validate_gesture_config(self) -> None:
        if self.gesture_stable_frames <= 0:
            raise ValueError(
                f"GESTURE_STABLE_FRAMES must be greater than 0, got {self.gesture_stable_frames}"
            )
        if self.gesture_cooldown_ms < 0:
            raise ValueError(
                "GESTURE_COOLDOWN_MS must be greater than or equal to 0, "
                f"got {self.gesture_cooldown_ms}"
            )
        if self.gesture_lost_hand_frames <= 0:
            raise ValueError(
                "GESTURE_LOST_HAND_FRAMES must be greater than 0, "
                f"got {self.gesture_lost_hand_frames}"
            )
        if self.clear_hold_ms <= 0:
            raise ValueError(f"CLEAR_HOLD_MS must be greater than 0, got {self.clear_hold_ms}")
        if not 0.0 <= self.finger_extension_margin < 1.0:
            raise ValueError(
                "FINGER_EXTENSION_MARGIN must be greater than or equal to 0.0 "
                f"and less than 1.0, got {self.finger_extension_margin}"
            )

    def validate_storage_config(self) -> None:
        if not str(self.drawing_output_dir).strip():
            raise ValueError("DRAWING_OUTPUT_DIR must not be empty")
        if self.drawing_image_format.lower().strip().lstrip(".") != "png":
            raise ValueError(f"DRAWING_IMAGE_FORMAT must be png, got {self.drawing_image_format!r}")
        if not self.drawing_filename_prefix.strip():
            raise ValueError("DRAWING_FILENAME_PREFIX must not be empty")
        if len(self.manual_save_key) != 1:
            raise ValueError("MANUAL_SAVE_KEY must contain exactly one character")
        if self.save_status_display_ms < 0:
            raise ValueError(
                "SAVE_STATUS_DISPLAY_MS must be greater than or equal to 0, "
                f"got {self.save_status_display_ms}"
            )

    def validate_preprocessing_config(self) -> None:
        if self.preprocess_output_width <= 0 or self.preprocess_output_height <= 0:
            raise ValueError(
                "PREPROCESS_OUTPUT_WIDTH and PREPROCESS_OUTPUT_HEIGHT must be greater than 0"
            )
        if self.preprocess_content_width <= 0 or self.preprocess_content_height <= 0:
            raise ValueError(
                "PREPROCESS_CONTENT_WIDTH and PREPROCESS_CONTENT_HEIGHT must be greater than 0"
            )
        if self.preprocess_content_width > self.preprocess_output_width:
            raise ValueError("PREPROCESS_CONTENT_WIDTH must be less than or equal to output width")
        if self.preprocess_content_height > self.preprocess_output_height:
            raise ValueError(
                "PREPROCESS_CONTENT_HEIGHT must be less than or equal to output height"
            )
        if not 0 <= self.preprocess_binary_threshold <= 255:
            raise ValueError("PREPROCESS_BINARY_THRESHOLD must be between 0 and 255")
        if self.preprocess_crop_padding < 0:
            raise ValueError("PREPROCESS_CROP_PADDING must be greater than or equal to 0")
        if self.preprocess_min_foreground_pixels <= 0:
            raise ValueError("PREPROCESS_MIN_FOREGROUND_PIXELS must be greater than 0")
        if len(self.manual_preprocess_key) != 1:
            raise ValueError("MANUAL_PREPROCESS_KEY must contain exactly one character")
        if self.manual_preprocess_key.lower() in {
            self.manual_save_key.lower(),
            self.canvas_clear_key.lower(),
        }:
            raise ValueError("MANUAL_PREPROCESS_KEY must not conflict with save or clear keys")
        if self.dataset_capture_image_format.lower().strip().lstrip(".") != "png":
            raise ValueError("DATASET_CAPTURE_IMAGE_FORMAT must be png")
        if not self.dataset_capture_filename_prefix.strip():
            raise ValueError("DATASET_CAPTURE_FILENAME_PREFIX must not be empty")
        writing_style = self.dataset_capture_writing_style.strip().lower()
        if writing_style not in {"uppercase", "lowercase"}:
            raise ValueError("DATASET_CAPTURE_WRITING_STYLE must be uppercase or lowercase")
        normalized_initial_label = self.dataset_capture_initial_label.strip()
        if (
            len(normalized_initial_label) != 1
            or normalized_initial_label.lower() not in "abcdefghijklmnopqrstuvwxyz"
        ):
            raise ValueError("DATASET_CAPTURE_INITIAL_LABEL must be one letter from A-Z or a-z")
        capture_keys = {
            self.dataset_capture_save_key.lower(),
            self.dataset_capture_next_label_key.lower(),
            self.dataset_capture_previous_label_key.lower(),
        }
        if any(len(key) != 1 for key in capture_keys):
            raise ValueError("Dataset capture keys must contain exactly one character")
        if len(capture_keys) != 3:
            raise ValueError("Dataset capture keys must be distinct")
        if self.dataset_capture_limit < 0:
            raise ValueError("DATASET_CAPTURE_LIMIT must be greater than or equal to zero")
        reserved_keys = {
            self.manual_save_key.lower(),
            self.manual_preprocess_key.lower(),
            self.canvas_clear_key.lower(),
            "d",
            "q",
        }
        if capture_keys & reserved_keys:
            raise ValueError(
                "Dataset capture keys must not conflict with save/preprocess/clear/done/quit keys"
            )

    def preprocessing_runtime_contract(self) -> dict[str, object]:
        return {
            "output_width": self.preprocess_output_width,
            "output_height": self.preprocess_output_height,
            "channels": 1,
            "background": "black",
            "foreground": "light",
            "normalized_min": 0.0,
            "normalized_max": 1.0,
            "normalization_divisor": preprocessing_alignment_settings.normalization_divisor,
            "source": "AIRWRITE_CANVAS",
            "orientation_transform": "none",
            "content_width": self.preprocess_content_width,
            "content_height": self.preprocess_content_height,
            "binary_threshold": self.preprocess_binary_threshold,
            "crop_padding": self.preprocess_crop_padding,
            "min_foreground_pixels": self.preprocess_min_foreground_pixels,
            "invert_input": self.preprocess_invert_input,
            "center_of_mass": self.preprocess_center_of_mass,
        }


@dataclass
class PreprocessingAlignmentSettings:
    input_width: int = int(os.getenv("MODEL_INPUT_WIDTH", "28"))
    input_height: int = int(os.getenv("MODEL_INPUT_HEIGHT", "28"))
    input_channels: int = int(os.getenv("MODEL_INPUT_CHANNELS", "1"))
    background_value: int = int(os.getenv("MODEL_INPUT_BACKGROUND_VALUE", "0"))
    normalization_divisor: float = env_to_float("MODEL_INPUT_NORMALIZATION_DIVISOR", 255.0)
    emnist_data_dir: Path = PROJECT_ROOT / os.getenv("EMNIST_DATA_DIR", "data/external/emnist")
    emnist_dataset_name: str = os.getenv("EMNIST_DATASET_NAME", "emnist/letters")
    emnist_transpose_images: bool = env_to_bool("EMNIST_TRANSPOSE_IMAGES", True)
    emnist_preserve_grayscale: bool = env_to_bool("EMNIST_PRESERVE_GRAYSCALE", True)
    audit_samples_per_class: int = int(os.getenv("PREPROCESS_AUDIT_SAMPLES_PER_CLASS", "10"))
    audit_output_dir: Path = PROJECT_ROOT / os.getenv(
        "PREPROCESS_AUDIT_OUTPUT_DIR", "artifacts/preprocessing_audit"
    )
    save_audit_images: bool = env_to_bool("SAVE_PREPROCESS_AUDIT_IMAGES", False)

    def validate(self) -> None:
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError("MODEL_INPUT_WIDTH and MODEL_INPUT_HEIGHT must be greater than 0")
        if self.input_channels != 1:
            raise ValueError("MODEL_INPUT_CHANNELS must be 1")
        if not 0 <= self.background_value <= 255:
            raise ValueError("MODEL_INPUT_BACKGROUND_VALUE must be between 0 and 255")
        if self.normalization_divisor <= 0.0:
            raise ValueError("MODEL_INPUT_NORMALIZATION_DIVISOR must be greater than 0")
        if not self.emnist_dataset_name.strip():
            raise ValueError("EMNIST_DATASET_NAME must not be empty")
        if not self.emnist_preserve_grayscale:
            raise ValueError("EMNIST_PRESERVE_GRAYSCALE must remain true in Sprint 8E")
        if self.audit_samples_per_class <= 0:
            raise ValueError("PREPROCESS_AUDIT_SAMPLES_PER_CLASS must be greater than 0")


@dataclass
class TrainingSettings:
    dataset_root: Path = PROJECT_ROOT / os.getenv("DATASET_ROOT", "data/raw_airwrite")
    manifest_path: Path = PROJECT_ROOT / os.getenv(
        "DATASET_MANIFEST_PATH", "data/manifests/dataset_manifest.csv"
    )
    split_path: Path = PROJECT_ROOT / os.getenv(
        "DATASET_SPLITS_PATH", "data/manifests/dataset_splits.csv"
    )
    train_ratio: float = env_to_float("TRAIN_RATIO", 0.70)
    validation_ratio: float = env_to_float("VALIDATION_RATIO", 0.15)
    test_ratio: float = env_to_float("TEST_RATIO", 0.15)
    random_seed: int = int(os.getenv("TRAINING_RANDOM_SEED", "42"))
    dataset_images_preprocessed: bool = env_to_bool("DATASET_IMAGES_PREPROCESSED", True)
    input_width: int = int(os.getenv("MODEL_INPUT_WIDTH", "28"))
    input_height: int = int(os.getenv("MODEL_INPUT_HEIGHT", "28"))
    input_channels: int = int(os.getenv("MODEL_INPUT_CHANNELS", "1"))
    num_classes: int = int(os.getenv("MODEL_NUM_CLASSES", "26"))
    batch_size: int = int(os.getenv("TRAIN_BATCH_SIZE", "64"))
    max_epochs: int = int(os.getenv("TRAIN_MAX_EPOCHS", "50"))
    learning_rate: float = env_to_float("TRAIN_LEARNING_RATE", 0.001)
    early_stopping_patience: int = int(os.getenv("TRAIN_EARLY_STOPPING_PATIENCE", "5"))
    enable_augmentation: bool = env_to_bool("TRAIN_ENABLE_AUGMENTATION", True)
    rotation_factor: float = env_to_float("TRAIN_ROTATION_FACTOR", 0.03)
    translation_factor: float = env_to_float("TRAIN_TRANSLATION_FACTOR", 0.08)
    zoom_factor: float = env_to_float("TRAIN_ZOOM_FACTOR", 0.08)
    model_output_path: Path = PROJECT_ROOT / os.getenv(
        "MODEL_OUTPUT_PATH", "artifacts/models/character_recognizer.keras"
    )
    labels_output_path: Path = PROJECT_ROOT / os.getenv(
        "LABELS_OUTPUT_PATH", "artifacts/labels/labels.json"
    )
    metrics_output_path: Path = PROJECT_ROOT / os.getenv(
        "METRICS_OUTPUT_PATH", "artifacts/metadata/metrics.json"
    )
    model_metadata_path: Path = PROJECT_ROOT / os.getenv(
        "MODEL_METADATA_PATH", "artifacts/metadata/model_metadata.json"
    )
    preprocessing_config_output_path: Path = PROJECT_ROOT / os.getenv(
        "PREPROCESSING_CONFIG_OUTPUT_PATH",
        "artifacts/metadata/preprocessing_config.json",
    )
    training_history_path: Path = PROJECT_ROOT / os.getenv(
        "TRAINING_HISTORY_PATH", "artifacts/reports/training_history.csv"
    )
    classification_report_path: Path = PROJECT_ROOT / os.getenv(
        "CLASSIFICATION_REPORT_PATH", "artifacts/reports/classification_report.json"
    )
    confusion_matrix_json_path: Path = PROJECT_ROOT / os.getenv(
        "CONFUSION_MATRIX_JSON_PATH", "artifacts/reports/confusion_matrix.json"
    )
    confusion_matrix_image_path: Path = PROJECT_ROOT / os.getenv(
        "CONFUSION_MATRIX_IMAGE_PATH", "artifacts/reports/confusion_matrix.png"
    )
    error_analysis_path: Path = PROJECT_ROOT / os.getenv(
        "ERROR_ANALYSIS_PATH", "artifacts/reports/error_analysis.csv"
    )
    prediction_probabilities_path: Path = PROJECT_ROOT / os.getenv(
        "PREDICTION_PROBABILITIES_PATH", "artifacts/reports/prediction_probabilities.csv"
    )
    experiment_log_path: Path = PROJECT_ROOT / os.getenv(
        "EXPERIMENT_LOG_PATH", "artifacts/reports/experiments.csv"
    )
    baseline_metrics_path: Path = PROJECT_ROOT / os.getenv(
        "BASELINE_METRICS_PATH", "artifacts/metadata/baseline_metrics.json"
    )
    model_name: str = os.getenv("MODEL_NAME", "airwrite_character_recognizer")
    model_version: str = os.getenv("MODEL_VERSION", "0.1.0")
    experiment_id: str = os.getenv("TRAIN_EXPERIMENT_ID", "exp_custom_001")
    experiment_notes: str = os.getenv("TRAIN_EXPERIMENT_NOTES", "Custom AirWrite uppercase A-Z")

    def validate(self) -> None:
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
            raise ValueError(
                "TRAIN_RATIO, VALIDATION_RATIO, and TEST_RATIO must be between 0 and 1"
            )
        if not isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO must equal 1.0")
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError("Model input dimensions must be greater than 0")
        if self.input_channels != 1:
            raise ValueError("MODEL_INPUT_CHANNELS must be 1 for grayscale AirWrite images")
        if self.num_classes != 26:
            raise ValueError("MODEL_NUM_CLASSES must match the 26 uppercase A-Z labels")
        if self.batch_size <= 0 or self.max_epochs <= 0:
            raise ValueError("TRAIN_BATCH_SIZE and TRAIN_MAX_EPOCHS must be greater than 0")
        if self.learning_rate <= 0.0:
            raise ValueError("TRAIN_LEARNING_RATE must be greater than 0")
        if self.early_stopping_patience < 0:
            raise ValueError("TRAIN_EARLY_STOPPING_PATIENCE must be greater than or equal to 0")
        if any(
            factor < 0.0 or factor >= 1.0
            for factor in (self.rotation_factor, self.translation_factor, self.zoom_factor)
        ):
            raise ValueError("Training augmentation factors must be between 0 inclusive and 1")
        if not self.model_name.strip() or not self.model_version.strip():
            raise ValueError("MODEL_NAME and MODEL_VERSION must not be empty")
        if not self.experiment_id.strip():
            raise ValueError("TRAIN_EXPERIMENT_ID must not be empty")

    def create_artifact_directories(self) -> None:
        artifact_paths = (
            self.model_output_path,
            self.labels_output_path,
            self.metrics_output_path,
            self.model_metadata_path,
            self.preprocessing_config_output_path,
            self.training_history_path,
            self.classification_report_path,
            self.confusion_matrix_json_path,
            self.confusion_matrix_image_path,
            self.error_analysis_path,
            self.prediction_probabilities_path,
            self.experiment_log_path,
            self.baseline_metrics_path,
        )
        for artifact_path in artifact_paths:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class EMNISTTrainingSettings:
    data_root: Path = PROJECT_ROOT / os.getenv(
        "EMNIST_OFFICIAL_DATA_DIR", "data/external/emnist/gzip"
    )
    artifact_root: Path = PROJECT_ROOT / os.getenv(
        "EMNIST_ARTIFACT_ROOT", "artifacts/emnist_letters_identity/v1"
    )
    airwrite_eval_manifest: Path = PROJECT_ROOT / os.getenv(
        "AIRWRITE_IDENTITY_EVAL_MANIFEST", "data/airwrite_identity_eval/manifest.csv"
    )
    source: str = os.getenv("EMNIST_SOURCE", "official_idx")
    raw_label_min: int = int(os.getenv("EMNIST_RAW_LABEL_MIN", "1"))
    raw_label_max: int = int(os.getenv("EMNIST_RAW_LABEL_MAX", "26"))
    validation_ratio: float = env_to_float("EMNIST_VALIDATION_RATIO", 0.10)
    random_seed: int = int(os.getenv("EMNIST_RANDOM_SEED", "42"))
    batch_size: int = int(os.getenv("EMNIST_BATCH_SIZE", "128"))
    shuffle_buffer: int = int(os.getenv("EMNIST_SHUFFLE_BUFFER", "20000"))
    cache_dataset: bool = env_to_bool("EMNIST_CACHE_DATASET", False)
    prefetch_dataset: bool = env_to_bool("EMNIST_PREFETCH_DATASET", True)
    max_epochs: int = int(os.getenv("EMNIST_MAX_EPOCHS", "50"))
    learning_rate: float = env_to_float("EMNIST_LEARNING_RATE", 0.001)
    early_stopping_patience: int = int(os.getenv("EMNIST_EARLY_STOPPING_PATIENCE", "5"))
    reduce_lr_patience: int = int(os.getenv("EMNIST_REDUCE_LR_PATIENCE", "3"))
    dropout_rate: float = env_to_float("EMNIST_DROPOUT_RATE", 0.30)
    rotation_factor: float = env_to_float("EMNIST_ROTATION_FACTOR", 0.03)
    translation_factor: float = env_to_float("EMNIST_TRANSLATION_FACTOR", 0.08)
    zoom_factor: float = env_to_float("EMNIST_ZOOM_FACTOR", 0.08)
    model_version: str = os.getenv("EMNIST_MODEL_VERSION", "1.0.0")

    @property
    def train_images_path(self) -> Path:
        return self.data_root / os.getenv(
            "EMNIST_TRAIN_IMAGES_FILE", "emnist-letters-train-images-idx3-ubyte.gz"
        )

    @property
    def train_labels_path(self) -> Path:
        return self.data_root / os.getenv(
            "EMNIST_TRAIN_LABELS_FILE", "emnist-letters-train-labels-idx1-ubyte.gz"
        )

    @property
    def test_images_path(self) -> Path:
        return self.data_root / os.getenv(
            "EMNIST_TEST_IMAGES_FILE", "emnist-letters-test-images-idx3-ubyte.gz"
        )

    @property
    def test_labels_path(self) -> Path:
        return self.data_root / os.getenv(
            "EMNIST_TEST_LABELS_FILE", "emnist-letters-test-labels-idx1-ubyte.gz"
        )

    @property
    def split_indices_path(self) -> Path:
        return self.artifact_root / "split_indices.npz"

    def experiment_root(self, experiment_id: str) -> Path:
        normalized = experiment_id.strip().upper()
        if normalized not in {"E01", "E02", "SMOKE"}:
            raise ValueError("experiment_id must be E01, E02, or SMOKE")
        return self.artifact_root / "experiments" / normalized

    def validate(self) -> None:
        if self.source != "official_idx":
            raise ValueError("EMNIST_SOURCE must be official_idx")
        if (self.raw_label_min, self.raw_label_max) != (1, 26):
            raise ValueError("EMNIST raw label contract must remain 1-26")
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("EMNIST_VALIDATION_RATIO must be between 0 and 1")
        if self.batch_size <= 0 or self.shuffle_buffer <= 0 or self.max_epochs <= 0:
            raise ValueError("EMNIST batch, shuffle buffer, and epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("EMNIST_LEARNING_RATE must be positive")
        if self.early_stopping_patience < 0 or self.reduce_lr_patience < 0:
            raise ValueError("EMNIST callback patience values must be non-negative")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("EMNIST_DROPOUT_RATE must be between 0 and 1")
        factors = (self.rotation_factor, self.translation_factor, self.zoom_factor)
        if any(factor < 0.0 or factor >= 1.0 for factor in factors):
            raise ValueError("EMNIST augmentation factors must be between 0 and 1")


@dataclass
class InferenceSettings:
    top_k: int = int(os.getenv("PREDICTION_TOP_K", "3"))
    min_confidence: float = env_to_float("PREDICTION_MIN_CONFIDENCE", 0.60)
    min_margin: float = env_to_float("PREDICTION_MIN_MARGIN", 0.15)
    auto_predict_on_done: bool = env_to_bool("ENABLE_AUTO_PREDICT_ON_DONE", True)
    manual_predict_enabled: bool = env_to_bool("ENABLE_MANUAL_PREDICT", True)
    manual_predict_key: str = os.getenv("MANUAL_PREDICT_KEY", "i")
    show_prediction_status: bool = env_to_bool("SHOW_PREDICTION_STATUS", True)
    show_top_k_predictions: bool = env_to_bool("SHOW_TOP_K_PREDICTIONS", True)
    status_display_ms: int = int(os.getenv("PREDICTION_STATUS_DISPLAY_MS", "4000"))
    log_latency: bool = env_to_bool("LOG_PREDICTION_LATENCY", True)
    latency_warning_ms: int = int(os.getenv("PREDICTION_LATENCY_WARNING_MS", "500"))

    def validate(self, num_classes: int = 26) -> None:
        if not 0 < self.top_k <= num_classes:
            raise ValueError(f"PREDICTION_TOP_K must be between 1 and {num_classes}")
        validate_confidence("PREDICTION_MIN_CONFIDENCE", self.min_confidence)
        validate_confidence("PREDICTION_MIN_MARGIN", self.min_margin)
        if len(self.manual_predict_key) != 1:
            raise ValueError("MANUAL_PREDICT_KEY must contain exactly one character")
        if self.status_display_ms < 0:
            raise ValueError("PREDICTION_STATUS_DISPLAY_MS must be greater than or equal to 0")
        if self.latency_warning_ms <= 0:
            raise ValueError("PREDICTION_LATENCY_WARNING_MS must be greater than 0")


@dataclass
class IdentityModelSettings:
    model_path: Path = PROJECT_ROOT / os.getenv(
        "IDENTITY_MODEL_PATH", "artifacts/airwrite_custom_identity/v2/model.keras"
    )
    identity_labels_path: Path = PROJECT_ROOT / os.getenv(
        "IDENTITY_LABELS_PATH", "artifacts/airwrite_custom_identity/v2/identity_labels.json"
    )
    lowercase_display_labels_path: Path = PROJECT_ROOT / os.getenv(
        "LOWERCASE_DISPLAY_LABELS_PATH",
        "artifacts/airwrite_custom_identity/v2/lowercase_display_labels.json",
    )
    uppercase_display_labels_path: Path = PROJECT_ROOT / os.getenv(
        "UPPERCASE_DISPLAY_LABELS_PATH",
        "artifacts/airwrite_custom_identity/v2/uppercase_display_labels.json",
    )
    metadata_path: Path = PROJECT_ROOT / os.getenv(
        "IDENTITY_MODEL_METADATA_PATH",
        "artifacts/airwrite_custom_identity/v2/model_metadata.json",
    )
    preprocessing_config_path: Path = PROJECT_ROOT / os.getenv(
        "IDENTITY_PREPROCESSING_CONFIG_PATH",
        "artifacts/airwrite_custom_identity/v2/preprocessing_config.json",
    )
    emnist_model_path: Path = PROJECT_ROOT / os.getenv(
        "EMNIST_SUPPORT_MODEL_PATH", "artifacts/emnist_letters_identity/v1/model.keras"
    )

    def validate(self) -> None:
        for name, path in (
            ("IDENTITY_MODEL_PATH", self.model_path),
            ("IDENTITY_LABELS_PATH", self.identity_labels_path),
            ("LOWERCASE_DISPLAY_LABELS_PATH", self.lowercase_display_labels_path),
            ("UPPERCASE_DISPLAY_LABELS_PATH", self.uppercase_display_labels_path),
            ("IDENTITY_MODEL_METADATA_PATH", self.metadata_path),
            ("IDENTITY_PREPROCESSING_CONFIG_PATH", self.preprocessing_config_path),
        ):
            if not str(path).strip():
                raise ValueError(f"{name} must not be empty")


@dataclass
class CaseControlSettings:
    default_mode: CharacterCaseMode = CharacterCaseMode.from_string(
        os.getenv("DEFAULT_CHARACTER_CASE_MODE", "lowercase")
    )
    lowercase_mode_key: str = os.getenv("LOWERCASE_MODE_KEY", "l")
    uppercase_mode_key: str = os.getenv("UPPERCASE_MODE_KEY", "u")
    shift_next_key: str = os.getenv("SHIFT_NEXT_KEY", "y")
    cancel_shift_key: str = os.getenv("CANCEL_SHIFT_KEY", "z")
    enable_auto_case_mode: bool = env_to_bool("ENABLE_AUTO_CASE_MODE", False)
    show_case_mode: bool = env_to_bool("SHOW_CHARACTER_CASE_MODE", True)
    show_case_debug: bool = env_to_bool("SHOW_CASE_INFERENCE_DEBUG", False)
    status_display_ms: int = int(os.getenv("CASE_STATUS_DISPLAY_MS", "3000"))

    def validate(self) -> None:
        keys = {
            "LOWERCASE_MODE_KEY": self.lowercase_mode_key,
            "UPPERCASE_MODE_KEY": self.uppercase_mode_key,
            "SHIFT_NEXT_KEY": self.shift_next_key,
            "CANCEL_SHIFT_KEY": self.cancel_shift_key,
        }
        if any(len(value) != 1 for value in keys.values()):
            raise ValueError("Case control keys must contain exactly one character")
        normalized = [value.lower() for value in keys.values()]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Case control keys must be distinct")
        if self.enable_auto_case_mode:
            raise ValueError("ENABLE_AUTO_CASE_MODE must remain false for the identity model")
        if self.status_display_ms < 0:
            raise ValueError("CASE_STATUS_DISPLAY_MS must be greater than or equal to 0")


@dataclass
class WordBuilderSettings:
    max_length: int = int(os.getenv("WORD_MAX_LENGTH", "30"))
    canonicalization: str = os.getenv("WORD_CANONICALIZATION", "casefold")
    auto_append_accepted: bool = env_to_bool("WORD_AUTO_APPEND_ACCEPTED", True)
    require_selection_for_uncertain: bool = env_to_bool(
        "WORD_REQUIRE_SELECTION_FOR_UNCERTAIN", True
    )
    candidate_1_key: str = os.getenv("WORD_SELECT_CANDIDATE_1_KEY", "1")
    candidate_2_key: str = os.getenv("WORD_SELECT_CANDIDATE_2_KEY", "2")
    candidate_3_key: str = os.getenv("WORD_SELECT_CANDIDATE_3_KEY", "3")
    cancel_pending_key: str = os.getenv("WORD_CANCEL_PENDING_KEY", "x")
    backspace_key: str = os.getenv("WORD_BACKSPACE_KEY", "b")
    clear_word_key: str = os.getenv("WORD_CLEAR_KEY", "k")
    confirm_key: str = os.getenv("WORD_CONFIRM_KEY", "enter")
    new_word_key: str = os.getenv("WORD_NEW_KEY", "n")
    auto_clear_after_commit: bool = env_to_bool("AUTO_CLEAR_CANVAS_AFTER_CHARACTER", True)
    auto_clear_after_pending_cancel: bool = env_to_bool(
        "AUTO_CLEAR_CANVAS_AFTER_PENDING_CANCEL", True
    )
    show_word_builder: bool = env_to_bool("SHOW_WORD_BUILDER", True)
    show_pending_candidates: bool = env_to_bool("SHOW_PENDING_CANDIDATES", True)
    status_display_ms: int = int(os.getenv("WORD_STATUS_DISPLAY_MS", "3000"))

    def validate(self) -> None:
        if self.max_length <= 0:
            raise ValueError("WORD_MAX_LENGTH must be greater than 0")
        if self.canonicalization.strip().lower() != "casefold":
            raise ValueError("WORD_CANONICALIZATION must be casefold")
        single_character_keys = {
            "WORD_SELECT_CANDIDATE_1_KEY": self.candidate_1_key,
            "WORD_SELECT_CANDIDATE_2_KEY": self.candidate_2_key,
            "WORD_SELECT_CANDIDATE_3_KEY": self.candidate_3_key,
            "WORD_CANCEL_PENDING_KEY": self.cancel_pending_key,
            "WORD_BACKSPACE_KEY": self.backspace_key,
            "WORD_CLEAR_KEY": self.clear_word_key,
            "WORD_NEW_KEY": self.new_word_key,
        }
        if any(len(value) != 1 for value in single_character_keys.values()):
            raise ValueError("Word Builder action keys must contain exactly one character")
        normalized_keys = [value.lower() for value in single_character_keys.values()]
        if len(set(normalized_keys)) != len(normalized_keys):
            raise ValueError("Word Builder action keys must be distinct")
        normalized_confirm = self.confirm_key.strip().lower()
        if normalized_confirm != "enter" and len(normalized_confirm) != 1:
            raise ValueError("WORD_CONFIRM_KEY must be 'enter' or one character")
        if normalized_confirm in normalized_keys:
            raise ValueError("WORD_CONFIRM_KEY must not conflict with another Word Builder key")
        if self.status_display_ms < 0:
            raise ValueError("WORD_STATUS_DISPLAY_MS must be greater than or equal to 0")


@dataclass(frozen=True)
class WholeWordSettings:
    default_input_mode: str = os.getenv("DEFAULT_DRAWING_INPUT_MODE", "character")
    character_mode_key: str = os.getenv("CHARACTER_MODE_KEY", "r")
    word_mode_key: str = os.getenv("WORD_MODE_KEY", "w")
    min_characters: int = int(os.getenv("WHOLE_WORD_MIN_CHARACTERS", "2"))
    max_characters: int = int(os.getenv("WHOLE_WORD_MAX_CHARACTERS", "12"))
    max_strokes: int = int(os.getenv("WHOLE_WORD_MAX_STROKES", "64"))
    max_points_per_stroke: int = int(os.getenv("WHOLE_WORD_MAX_POINTS_PER_STROKE", "1000"))
    roi_x_ratio: float = env_to_float("WORD_ROI_X_RATIO", 0.05)
    roi_y_ratio: float = env_to_float("WORD_ROI_Y_RATIO", 0.20)
    roi_width_ratio: float = env_to_float("WORD_ROI_WIDTH_RATIO", 0.90)
    roi_height_ratio: float = env_to_float("WORD_ROI_HEIGHT_RATIO", 0.60)
    min_foreground_pixels: int = int(os.getenv("WORD_SEGMENT_MIN_FOREGROUND_PIXELS", "10"))
    min_component_area: int = int(os.getenv("WORD_SEGMENT_MIN_COMPONENT_AREA", "4"))
    min_separator_gap: int = int(os.getenv("WORD_SEGMENT_MIN_SEPARATOR_GAP", "10"))
    max_internal_gap: int = int(os.getenv("WORD_SEGMENT_MAX_INTERNAL_GAP", "16"))
    x_overlap_threshold: float = env_to_float("WORD_SEGMENT_X_OVERLAP_THRESHOLD", 0.25)
    tiny_component_ratio: float = env_to_float("WORD_SEGMENT_TINY_COMPONENT_RATIO", 0.20)
    wide_group_ratio: float = env_to_float("WORD_SEGMENT_WIDE_GROUP_RATIO", 1.80)
    top_k: int = int(os.getenv("WHOLE_WORD_TOP_K", "3"))
    min_confidence: float = env_to_float("WHOLE_WORD_MIN_CONFIDENCE", 0.60)
    min_margin: float = env_to_float("WHOLE_WORD_MIN_MARGIN", 0.15)
    default_case_policy: str = os.getenv("DEFAULT_WORD_CASE_POLICY", "lowercase")
    case_lowercase_key: str = os.getenv("WORD_CASE_LOWERCASE_KEY", "l")
    case_uppercase_key: str = os.getenv("WORD_CASE_UPPERCASE_KEY", "u")
    case_capitalize_key: str = os.getenv("WORD_CASE_CAPITALIZE_KEY", "p")
    case_custom_key: str = os.getenv("WORD_CASE_CUSTOM_KEY", "o")
    previous_position_key: str = os.getenv("WORD_PREVIOUS_POSITION_KEY", "[")
    next_position_key: str = os.getenv("WORD_NEXT_POSITION_KEY", "]")
    toggle_case_key: str = os.getenv("WORD_TOGGLE_SELECTED_CASE_KEY", "g")
    split_segment_key: str = os.getenv("WORD_SPLIT_SELECTED_SEGMENT_KEY", "s")
    merge_segment_key: str = os.getenv("WORD_MERGE_SELECTED_WITH_NEXT_KEY", "v")
    accept_draft_key: str = os.getenv("WORD_ACCEPT_DRAFT_KEY", "a")
    cancel_draft_key: str = os.getenv("WORD_CANCEL_DRAFT_KEY", "x")
    audit_dir: Path = PROJECT_ROOT / os.getenv(
        "WORD_SEGMENT_AUDIT_DIR", "artifacts/word_segmentation_audit"
    )
    save_audit_images: bool = env_to_bool("SAVE_WORD_SEGMENT_AUDIT_IMAGES", False)
    show_roi: bool = env_to_bool("SHOW_WORD_ROI", True)
    show_segment_boxes: bool = env_to_bool("SHOW_WORD_SEGMENT_BOXES", True)
    show_segmentation_confidence: bool = env_to_bool("SHOW_WORD_SEGMENT_CONFIDENCE", False)
    show_prediction_confidence: bool = env_to_bool("SHOW_WORD_PREDICTION_CONFIDENCE", True)

    def validate(self) -> None:
        if self.default_input_mode.strip().upper() not in {"CHARACTER", "ISOLATED_WORD", "WORD"}:
            raise ValueError("DEFAULT_DRAWING_INPUT_MODE must be character or isolated_word")
        if not 2 <= self.min_characters <= self.max_characters <= 26:
            raise ValueError("Whole-word character limits must satisfy 2 <= min <= max <= 26")
        if self.max_strokes <= 0 or self.max_points_per_stroke <= 0:
            raise ValueError("Whole-word stroke and point limits must be positive")
        ratios = (
            self.roi_x_ratio,
            self.roi_y_ratio,
            self.roi_width_ratio,
            self.roi_height_ratio,
        )
        if any(value < 0.0 or value > 1.0 for value in ratios):
            raise ValueError("Word ROI ratios must be between 0 and 1")
        if self.roi_width_ratio <= 0.0 or self.roi_height_ratio <= 0.0:
            raise ValueError("Word ROI width and height ratios must be positive")
        if self.roi_x_ratio + self.roi_width_ratio > 1.0:
            raise ValueError("Word ROI exceeds frame width")
        if self.roi_y_ratio + self.roi_height_ratio > 1.0:
            raise ValueError("Word ROI exceeds frame height")
        positive_values = (
            self.min_foreground_pixels,
            self.min_component_area,
            self.min_separator_gap,
            self.max_internal_gap,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError("Whole-word segmentation pixel settings must be positive")
        if not 0.0 <= self.x_overlap_threshold <= 1.0:
            raise ValueError("WORD_SEGMENT_X_OVERLAP_THRESHOLD must be between 0 and 1")
        if not 0.0 < self.tiny_component_ratio <= 1.0:
            raise ValueError("WORD_SEGMENT_TINY_COMPONENT_RATIO must be between 0 and 1")
        if self.wide_group_ratio <= 1.0:
            raise ValueError("WORD_SEGMENT_WIDE_GROUP_RATIO must be greater than 1")
        if not 1 <= self.top_k <= 26:
            raise ValueError("WHOLE_WORD_TOP_K must be between 1 and 26")
        validate_confidence("WHOLE_WORD_MIN_CONFIDENCE", self.min_confidence)
        validate_confidence("WHOLE_WORD_MIN_MARGIN", self.min_margin)
        keys = (
            self.character_mode_key,
            self.word_mode_key,
            self.case_lowercase_key,
            self.case_uppercase_key,
            self.case_capitalize_key,
            self.case_custom_key,
            self.previous_position_key,
            self.next_position_key,
            self.toggle_case_key,
            self.split_segment_key,
            self.merge_segment_key,
            self.accept_draft_key,
            self.cancel_draft_key,
        )
        if any(len(key) != 1 for key in keys):
            raise ValueError("Whole-word control keys must contain one character")


settings = Settings()
preprocessing_alignment_settings = PreprocessingAlignmentSettings()
training_settings = TrainingSettings()
emnist_training_settings = EMNISTTrainingSettings()
inference_settings = InferenceSettings()
identity_model_settings = IdentityModelSettings()
case_control_settings = CaseControlSettings()
word_builder_settings = WordBuilderSettings()
whole_word_settings = WholeWordSettings()
