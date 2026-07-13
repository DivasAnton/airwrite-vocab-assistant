import cv2
import numpy as np
from numpy.typing import NDArray

from app.vision.hand_detection_result import DetectedHand, HandDetectionResult

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


class HandLandmarkRenderer:
    def __init__(
        self,
        draw_landmarks: bool = True,
        draw_connections: bool = True,
        draw_handedness: bool = True,
    ) -> None:
        self.draw_landmarks = draw_landmarks
        self.draw_connections = draw_connections
        self.draw_handedness = draw_handedness

    def draw(self, frame: NDArray[np.uint8], result: HandDetectionResult) -> NDArray[np.uint8]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be a non-empty color image")
        if frame.size == 0:
            raise ValueError("frame must be a non-empty color image")

        output = frame.copy()
        for hand in result.hands:
            points = self._landmarks_to_pixels(hand, output.shape[1], output.shape[0])
            if self.draw_connections:
                self._draw_connections(output, points)
            if self.draw_landmarks:
                self._draw_landmarks(output, points)
            if self.draw_handedness:
                self._draw_handedness(output, hand, points)

        return output

    @staticmethod
    def _landmarks_to_pixels(hand: DetectedHand, width: int, height: int) -> list[tuple[int, int]]:
        points = []
        for landmark in hand.normalized_landmarks:
            x = min(max(int(landmark.x * width), 0), width - 1)
            y = min(max(int(landmark.y * height), 0), height - 1)
            points.append((x, y))
        return points

    @staticmethod
    def _draw_connections(frame: NDArray[np.uint8], points: list[tuple[int, int]]) -> None:
        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (0, 180, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _draw_landmarks(frame: NDArray[np.uint8], points: list[tuple[int, int]]) -> None:
        for point in points:
            cv2.circle(frame, point, 4, (0, 255, 0), -1, cv2.LINE_AA)

    @staticmethod
    def _draw_handedness(
        frame: NDArray[np.uint8], hand: DetectedHand, points: list[tuple[int, int]]
    ) -> None:
        if not points:
            return

        label = hand.handedness or "Unknown"
        if hand.handedness_score is not None:
            label = f"{label} {hand.handedness_score:.2f}"

        x, y = points[0]
        cv2.putText(
            frame,
            label,
            (x, max(y - 12, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
