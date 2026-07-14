import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

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
    preprocess_output_width: int = int(os.getenv("PREPROCESS_OUTPUT_WIDTH", "28"))
    preprocess_output_height: int = int(os.getenv("PREPROCESS_OUTPUT_HEIGHT", "28"))
    preprocess_content_width: int = int(os.getenv("PREPROCESS_CONTENT_WIDTH", "20"))
    preprocess_content_height: int = int(os.getenv("PREPROCESS_CONTENT_HEIGHT", "20"))
    preprocess_binary_threshold: int = int(os.getenv("PREPROCESS_BINARY_THRESHOLD", "20"))
    preprocess_crop_padding: int = int(os.getenv("PREPROCESS_CROP_PADDING", "8"))
    preprocess_min_foreground_pixels: int = int(os.getenv("PREPROCESS_MIN_FOREGROUND_PIXELS", "10"))
    preprocess_invert_input: bool = env_to_bool("PREPROCESS_INVERT_INPUT", False)
    preprocess_center_of_mass: bool = env_to_bool("PREPROCESS_CENTER_OF_MASS", False)
    save_preprocess_debug_images: bool = env_to_bool("SAVE_PREPROCESS_DEBUG_IMAGES", False)
    preprocess_debug_output_dir: Path = PROJECT_ROOT / os.getenv(
        "PREPROCESS_DEBUG_OUTPUT_DIR", "data/preprocessed_debug"
    )
    manual_preprocess_key: str = os.getenv("MANUAL_PREPROCESS_KEY", "p")

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


settings = Settings()
