from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.drawing.air_canvas import AirCanvas
from app.drawing.drawing_state import DrawingState
from app.storage.drawing_image_saver import DrawingImageSaver
from app.storage.drawing_save_coordinator import DrawingSaveCoordinator
from app.storage.exceptions import DrawingImageSaveError
from app.storage.save_result import SaveResult, SaveStatus


class FakeSaver:
    def __init__(self) -> None:
        self.calls = 0
        self.saved_images: list[NDArray[np.uint8]] = []

    def save(self, image: NDArray[np.uint8]) -> SaveResult:
        self.calls += 1
        self.saved_images.append(image)
        return SaveResult(
            status=SaveStatus.SAVED,
            file_path=Path(f"drawing_{self.calls}.png"),
            saved_at=datetime.now(),
            message="saved",
        )


class FailingSaver:
    def save(self, image: NDArray[np.uint8]) -> SaveResult:
        raise DrawingImageSaveError("save failed")


def make_canvas(has_content: bool = True) -> AirCanvas:
    canvas = AirCanvas(width=80, height=60)
    if has_content:
        canvas.draw_line((5, 5), (50, 45))
    return canvas


def test_handle_state_ignores_non_done_states() -> None:
    saver = FakeSaver()
    coordinator = DrawingSaveCoordinator(saver=saver)

    result = coordinator.handle_state(DrawingState.WRITING, make_canvas())

    assert result is None
    assert saver.calls == 0


def test_handle_state_saves_once_when_entering_done() -> None:
    saver = FakeSaver()
    coordinator = DrawingSaveCoordinator(saver=saver)
    canvas = make_canvas()

    coordinator.handle_state(DrawingState.WRITING, canvas)
    first = coordinator.handle_state(DrawingState.DONE, canvas)
    second = coordinator.handle_state(DrawingState.DONE, canvas)

    assert first is not None
    assert first.status == SaveStatus.SAVED
    assert second is None
    assert saver.calls == 1


def test_handle_state_skips_empty_canvas_on_done() -> None:
    saver = FakeSaver()
    coordinator = DrawingSaveCoordinator(saver=saver)

    result = coordinator.handle_state(DrawingState.DONE, make_canvas(has_content=False))

    assert result is not None
    assert result.status == SaveStatus.SKIPPED_EMPTY
    assert saver.calls == 0


def test_handle_state_reports_disabled_when_auto_save_is_off() -> None:
    saver = FakeSaver()
    coordinator = DrawingSaveCoordinator(saver=saver, auto_save_on_done=False)

    result = coordinator.handle_state(DrawingState.DONE, make_canvas())

    assert result is not None
    assert result.status == SaveStatus.DISABLED
    assert saver.calls == 0


def test_manual_save_writes_even_when_auto_save_is_off() -> None:
    saver = FakeSaver()
    coordinator = DrawingSaveCoordinator(saver=saver, auto_save_on_done=False)

    result = coordinator.save_now(make_canvas())

    assert result.status == SaveStatus.SAVED
    assert saver.calls == 1


def test_manual_save_can_be_disabled() -> None:
    saver = FakeSaver()
    coordinator = DrawingSaveCoordinator(
        saver=saver,
        auto_save_on_done=False,
        enable_manual_save=False,
    )

    result = coordinator.save_now(make_canvas())

    assert result.status == SaveStatus.DISABLED
    assert saver.calls == 0


def test_save_failure_returns_failed_result() -> None:
    coordinator = DrawingSaveCoordinator(saver=FailingSaver())  # type: ignore[arg-type]

    result = coordinator.handle_state(DrawingState.DONE, make_canvas())

    assert result is not None
    assert result.status == SaveStatus.FAILED
    assert "save failed" in result.message


def test_save_uses_canvas_snapshot_copy() -> None:
    saver = FakeSaver()
    coordinator = DrawingSaveCoordinator(saver=saver)
    canvas = make_canvas()

    coordinator.handle_state(DrawingState.DONE, canvas)
    canvas.clear()

    assert saver.saved_images
    assert np.any(saver.saved_images[0] != canvas.background_color)


def test_integration_writes_one_png_for_done_transition(tmp_path: Path) -> None:
    saver = DrawingImageSaver(output_dir=tmp_path)
    coordinator = DrawingSaveCoordinator(saver=saver)
    canvas = make_canvas()

    first = coordinator.handle_state(DrawingState.DONE, canvas)
    second = coordinator.handle_state(DrawingState.DONE, canvas)

    assert first is not None
    assert first.status == SaveStatus.SAVED
    assert first.file_path is not None
    assert first.file_path.exists()
    assert second is None
    assert len(list(tmp_path.glob("*.png"))) == 1
