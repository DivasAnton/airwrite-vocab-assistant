import numpy as np

from app.word_recognition.input_mode_controller import InputModeController
from app.word_recognition.stroke_point import StrokePoint
from app.word_recognition.stroke_recorder import StrokeRecorder
from app.word_recognition.word_input_snapshot import WordInputSnapshot
from app.word_recognition.word_writing_region import WordWritingRegion


def test_word_writing_region_uses_canvas_ratios_and_bounds() -> None:
    region = WordWritingRegion.from_ratios(
        1000,
        500,
        x_ratio=0.05,
        y_ratio=0.20,
        width_ratio=0.90,
        height_ratio=0.60,
    )

    assert region == WordWritingRegion(50, 100, 900, 300)
    assert region.contains((50, 100))
    assert region.contains((949, 399))
    assert not region.contains((950, 399))


def test_stroke_recorder_closes_strokes_and_snapshot_is_immutable() -> None:
    recorder = StrokeRecorder(stroke_id_factory=lambda: "stroke-1")
    recorder.start_stroke(StrokePoint(10, 20, 1))
    recorder.append_point(StrokePoint(11, 21, 2))
    recorder.append_point(StrokePoint(20, 25, 3))
    stroke = recorder.end_stroke()

    assert stroke is not None
    assert stroke.bounding_box.width == 11
    assert stroke.bounding_box.height == 6
    canvas = np.zeros((100, 200, 3), dtype=np.uint8)
    snapshot = WordInputSnapshot(
        snapshot_id="snapshot-1",
        canvas_image=canvas,
        strokes=recorder.snapshot(),
        writing_region=WordWritingRegion(0, 0, 200, 100),
        completed_at_ms=4,
    )
    canvas[20, 10] = 255

    assert not snapshot.canvas_image.flags.writeable
    assert snapshot.canvas_image[20, 10].sum() == 0
