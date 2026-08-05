from datetime import datetime

import numpy as np
from numpy.typing import NDArray

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
        return self.save_snapshot(
            canvas.get_image(copy=True),
            is_empty=canvas.is_empty(),
            manual=True,
        )

    def save_snapshot(
        self,
        snapshot: NDArray[np.uint8],
        *,
        is_empty: bool,
        manual: bool,
    ) -> SaveResult:
        if manual and not self.enable_manual_save:
            return SaveResult(
                status=SaveStatus.DISABLED,
                file_path=None,
                saved_at=None,
                message="Manual save is disabled",
            )
        if not manual and not self.auto_save_on_done:
            return SaveResult(
                status=SaveStatus.DISABLED,
                file_path=None,
                saved_at=None,
                message="Auto-save is disabled",
            )
        if is_empty:
            return SaveResult(
                status=SaveStatus.SKIPPED_EMPTY,
                file_path=None,
                saved_at=None,
                message="Skipped save because canvas is empty",
            )
        return self._save_snapshot(snapshot)

    def _save_canvas(self, canvas: AirCanvas) -> SaveResult:
        if canvas.is_empty():
            return SaveResult(
                status=SaveStatus.SKIPPED_EMPTY,
                file_path=None,
                saved_at=None,
                message="Skipped save because canvas is empty",
            )

        snapshot = canvas.get_image(copy=True)
        return self._save_snapshot(snapshot)

    def _save_snapshot(self, snapshot: NDArray[np.uint8]) -> SaveResult:
        try:
            return self.saver.save(snapshot)
        except DrawingStorageError as error:
            return SaveResult(
                status=SaveStatus.FAILED,
                file_path=None,
                saved_at=datetime.now(),
                message=str(error),
            )
