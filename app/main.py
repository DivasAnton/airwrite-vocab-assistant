from time import perf_counter

import cv2

from app.camera.camera_stream import CameraStream
from app.camera.frame_processor import FrameProcessor
from app.drawing.air_canvas import AirCanvas
from app.drawing.canvas_overlay_renderer import CanvasOverlayRenderer
from app.drawing.stroke_manager import StrokeManager
from app.utils.config import settings
from app.utils.logger import get_logger
from app.vision.finger_tracking_renderer import FingerTrackingRenderer
from app.vision.hand_detector import HandDetector
from app.vision.hand_landmark_renderer import HandLandmarkRenderer
from app.vision.index_finger_tracker import IndexFingerTracker

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


def create_air_canvas(width: int, height: int) -> AirCanvas:
    return AirCanvas(
        width=width,
        height=height,
        background_color=settings.canvas_background_color,
        stroke_color=settings.canvas_stroke_color,
        stroke_thickness=settings.canvas_stroke_thickness,
    )


def main() -> None:
    settings.create_directories()
    settings.validate_hand_detection_config()
    settings.validate_finger_tracking_config()
    settings.validate_canvas_config()

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
        draw_handedness=settings.draw_handedness,
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
    stroke_manager = StrokeManager(max_point_distance=settings.canvas_max_point_distance)
    overlay_renderer = CanvasOverlayRenderer(
        background_color=settings.canvas_background_color,
        opacity=settings.canvas_overlay_opacity,
    )
    air_canvas: AirCanvas | None = None
    previous_timestamp_ms = 0

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
            elif not air_canvas.matches_size(width=frame_width, height=frame_height):
                air_canvas.reset_size(width=frame_width, height=frame_height)
                stroke_manager.reset()

            previous_timestamp_ms = next_timestamp_ms(previous_timestamp_ms)
            hand_result = detector.detect(mirrored_frame, previous_timestamp_ms)
            finger_result = finger_tracker.track(
                hand_result,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            segment = stroke_manager.update(finger_result.smoothed_point)
            if segment is not None:
                air_canvas.draw_line(segment.start, segment.end)

            annotated_frame = renderer.draw(mirrored_frame, hand_result)
            annotated_frame = finger_renderer.draw(annotated_frame, finger_result)
            canvas_image = air_canvas.get_image()
            if settings.show_camera_with_canvas:
                annotated_frame = overlay_renderer.render(annotated_frame, canvas_image)

            fps = processor.calculate_fps()
            hand_label = (
                hand_result.hands[0].handedness
                if hand_result.has_hands and hand_result.hands[0].handedness
                else "None"
            )
            finger_status = "TRACKED" if finger_result.is_tracked else "NOT DETECTED"
            finger_point = (
                f"{finger_result.smoothed_point[0]}, {finger_result.smoothed_point[1]}"
                if finger_result.smoothed_point is not None
                else "None"
            )
            display_frame = processor.draw_debug_info(
                annotated_frame,
                fps=fps,
                show_fps=settings.show_fps,
                extra_lines=[
                    f"Hands: {len(hand_result.hands)}",
                    f"Hand: {hand_label}",
                    f"Detection: {'Active' if hand_result.has_hands else 'No hand detected'}",
                    f"Finger: {finger_status}",
                    f"Point: {finger_point}",
                    f"Canvas: {'Empty' if air_canvas.is_empty() else 'Drawing'}",
                    f"Clear: {settings.canvas_clear_key.upper()}",
                ],
            )

            cv2.imshow(settings.camera_window_name, display_frame)
            if settings.show_canvas_window:
                cv2.imshow("AirWrite Canvas", canvas_image)

            key = cv2.waitKey(1) & 0xFF
            if air_canvas is not None and is_clear_key(key, settings.canvas_clear_key):
                air_canvas.clear()
                stroke_manager.reset()
                logger.info("Canvas cleared")
                continue
            if should_quit(settings.camera_window_name, key):
                logger.info("Camera loop stopped by user")
                break
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
    finally:
        if detector is not None:
            detector.close()
        finger_tracker.reset()
        stroke_manager.reset()
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Camera and hand detector resources released")


if __name__ == "__main__":
    main()
