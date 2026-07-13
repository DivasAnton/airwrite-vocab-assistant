from app.vision.finger_tracking_result import FingerTrackingResult
from app.vision.hand_detection_result import DetectedHand, HandDetectionResult

INDEX_FINGER_TIP_LANDMARK_INDEX = 8


class IndexFingerTracker:
    def __init__(
        self,
        landmark_index: int = INDEX_FINGER_TIP_LANDMARK_INDEX,
        smoothing_alpha: float = 0.5,
    ) -> None:
        if not 0 <= landmark_index <= 20:
            raise ValueError(f"landmark_index must be between 0 and 20, got {landmark_index}")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError(f"smoothing_alpha must be in (0.0, 1.0], got {smoothing_alpha}")

        self.landmark_index = landmark_index
        self.smoothing_alpha = smoothing_alpha
        self.previous_smoothed_x: float | None = None
        self.previous_smoothed_y: float | None = None

    def track(
        self,
        result: HandDetectionResult,
        frame_width: int,
        frame_height: int,
    ) -> FingerTrackingResult:
        self._validate_frame_dimensions(frame_width, frame_height)
        hand = self._select_hand(result)
        if hand is None or len(hand.normalized_landmarks) <= self.landmark_index:
            self.reset()
            return self._not_tracked(result.timestamp_ms)

        landmark = hand.normalized_landmarks[self.landmark_index]
        raw_point = self._normalized_to_pixel(landmark.x, landmark.y, frame_width, frame_height)
        smoothed_point = self._smooth(raw_point)

        return FingerTrackingResult(
            raw_point=raw_point,
            smoothed_point=smoothed_point,
            normalized_point=(landmark.x, landmark.y),
            is_tracked=True,
            timestamp_ms=result.timestamp_ms,
            hand_index=0,
            landmark_index=self.landmark_index,
        )

    def reset(self) -> None:
        self.previous_smoothed_x = None
        self.previous_smoothed_y = None

    @staticmethod
    def _validate_frame_dimensions(frame_width: int, frame_height: int) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(
                "frame_width and frame_height must be greater than 0, "
                f"got {frame_width}x{frame_height}"
            )

    @staticmethod
    def _select_hand(result: HandDetectionResult) -> DetectedHand | None:
        if not result.hands:
            return None
        return result.hands[0]

    @staticmethod
    def _normalized_to_pixel(
        normalized_x: float,
        normalized_y: float,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int]:
        raw_x = int(normalized_x * frame_width)
        raw_y = int(normalized_y * frame_height)
        pixel_x = min(max(raw_x, 0), frame_width - 1)
        pixel_y = min(max(raw_y, 0), frame_height - 1)
        return pixel_x, pixel_y

    def _smooth(self, raw_point: tuple[int, int]) -> tuple[int, int]:
        raw_x, raw_y = raw_point
        if self.previous_smoothed_x is None or self.previous_smoothed_y is None:
            self.previous_smoothed_x = float(raw_x)
            self.previous_smoothed_y = float(raw_y)
            return raw_point

        smoothed_x = (
            self.smoothing_alpha * raw_x + (1 - self.smoothing_alpha) * self.previous_smoothed_x
        )
        smoothed_y = (
            self.smoothing_alpha * raw_y + (1 - self.smoothing_alpha) * self.previous_smoothed_y
        )
        self.previous_smoothed_x = smoothed_x
        self.previous_smoothed_y = smoothed_y
        return round(smoothed_x), round(smoothed_y)

    def _not_tracked(self, timestamp_ms: int) -> FingerTrackingResult:
        return FingerTrackingResult(
            raw_point=None,
            smoothed_point=None,
            normalized_point=None,
            is_tracked=False,
            timestamp_ms=timestamp_ms,
            hand_index=None,
            landmark_index=self.landmark_index,
        )
