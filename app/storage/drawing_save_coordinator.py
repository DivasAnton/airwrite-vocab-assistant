from datetime import datetime

from app.drawing.air_canvas import AirCanvas
from app.drawing.drawing_state import DrawingState
from app.storage.drawing_image_saver import DrawingImageSaver
from app.storage.exceptions import DrawingStorageError
from app.storage.save_result import SaveResult, SaveStatus


class DrawingSaveCoordinator:
    def __init__(
        self,
        saver: DrawingImageSaver,
        auto_save_on_done: bool = True,
        enable_manual_save: bool = True,
    ) -> None:
        self.saver = saver
        self.auto_save_on_done = auto_save_on_done
        self.enable_manual_save = enable_manual_save
        self.previous_state = DrawingState.IDLE

    def handle_state(self, current_state: DrawingState, canvas: AirCanvas) -> SaveResult | None:
        entered_done = (
            self.previous_state != DrawingState.DONE and current_state == DrawingState.DONE
        )
        self.previous_state = current_state
        if not entered_done:
            return None
        if not self.auto_save_on_done:
            return SaveResult(
                status=SaveStatus.DISABLED,
                file_path=None,
                saved_at=None,
                message="Auto-save is disabled",
            )
        return self._save_canvas(canvas)

    def save_now(self, canvas: AirCanvas) -> SaveResult:
        if not self.enable_manual_save:
            return SaveResult(
                status=SaveStatus.DISABLED,
                file_path=None,
                saved_at=None,
                message="Manual save is disabled",
            )
        return self._save_canvas(canvas)

    def _save_canvas(self, canvas: AirCanvas) -> SaveResult:
        if canvas.is_empty():
            return SaveResult(
                status=SaveStatus.SKIPPED_EMPTY,
                file_path=None,
                saved_at=None,
                message="Skipped save because canvas is empty",
            )

        snapshot = canvas.get_image(copy=True)
        try:
            return self.saver.save(snapshot)
        except DrawingStorageError as error:
            return SaveResult(
                status=SaveStatus.FAILED,
                file_path=None,
                saved_at=datetime.now(),
                message=str(error),
            )
