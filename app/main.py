import cv2

from app.camera.camera_stream import CameraStream
from app.camera.frame_processor import FrameProcessor
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def should_quit(window_name: str, key: int) -> bool:
    if key in {27, ord("q"), ord("Q")}:
        return True

    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main() -> None:
    settings.create_directories()

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

    try:
        camera.open()
        cv2.namedWindow(settings.camera_window_name, cv2.WINDOW_NORMAL)

        while True:
            success, frame = camera.read()
            if not success or not processor.validate_frame(frame):
                logger.error("Stopping camera loop because frame is invalid")
                break

            assert frame is not None
            mirrored_frame = processor.mirror_frame(frame, settings.camera_mirror)
            fps = processor.calculate_fps()
            display_frame = processor.draw_debug_info(
                mirrored_frame,
                fps=fps,
                show_fps=settings.show_fps,
            )

            cv2.imshow(settings.camera_window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if should_quit(settings.camera_window_name, key):
                logger.info("Camera loop stopped by user")
                break
    except RuntimeError as error:
        logger.error("%s", error)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Camera resources released")


if __name__ == "__main__":
    main()
