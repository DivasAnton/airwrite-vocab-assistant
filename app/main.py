from time import perf_counter

import cv2

from app.camera.camera_stream import CameraStream
from app.camera.frame_processor import FrameProcessor
from app.drawing.air_canvas import AirCanvas
from app.drawing.canvas_overlay_renderer import CanvasOverlayRenderer
from app.drawing.drawing_controller import DrawingController
from app.drawing.drawing_state import DrawingState
from app.drawing.drawing_state_machine import DrawingStateMachine
from app.drawing.stroke_manager import StrokeManager
from app.utils.config import settings
from app.utils.logger import get_logger
from app.vision.finger_tracking_renderer import FingerTrackingRenderer
from app.vision.gesture import Gesture
from app.vision.gesture_detector import GestureDetector
from app.vision.gesture_stabilizer import GestureStabilizer
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


def is_done_key(key: int) -> bool:
    return key in {ord("d"), ord("D")}


def is_space_key(key: int) -> bool:
    return key == ord(" ")


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
    settings.validate_gesture_config()

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
    air_canvas: AirCanvas | None = None
    drawing_controller: DrawingController | None = None
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
                drawing_controller = DrawingController(state_machine, stroke_manager, air_canvas)
            elif not air_canvas.matches_size(width=frame_width, height=frame_height):
                air_canvas.reset_size(width=frame_width, height=frame_height)
                stroke_manager.reset()
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
            controller_result = drawing_controller.update(stable_gesture, finger_result)

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
                    *(
                        [
                            f"Raw Gesture: {raw_gesture.value}",
                            f"Stable Gesture: {stable_gesture.value}",
                            (
                                "Candidate: "
                                f"{gesture_stabilizer.candidate_gesture.value} "
                                f"{gesture_stabilizer.candidate_frame_count}/"
                                f"{settings.gesture_stable_frames}"
                            ),
                        ]
                        if settings.draw_gesture_label
                        else []
                    ),
                    *(
                        [f"State: {controller_result.state.value}"]
                        if settings.draw_state_label
                        else []
                    ),
                ],
            )

            cv2.imshow(settings.camera_window_name, display_frame)
            if settings.show_canvas_window:
                cv2.imshow("AirWrite Canvas", canvas_image)

            key = cv2.waitKey(1) & 0xFF
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
                drawing_controller.set_state(DrawingState.DONE, previous_timestamp_ms)
                logger.info("Keyboard fallback changed state to DONE")
                continue
            if (
                drawing_controller is not None
                and is_clear_key(key, settings.canvas_clear_key)
                and settings.enable_keyboard_fallback
            ):
                drawing_controller.clear(previous_timestamp_ms)
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
        gesture_stabilizer.reset()
        finger_tracker.reset()
        stroke_manager.reset()
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Camera and hand detector resources released")


if __name__ == "__main__":
    main()
