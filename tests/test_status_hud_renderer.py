import numpy as np

from app.status_hud_renderer import StatusHudRenderer


def test_hud_renders_only_inside_top_right_panel() -> None:
    renderer = StatusHudRenderer(target_font_height_px=13, margin_px=10)
    frame = np.zeros((240, 640, 3), dtype=np.uint8)

    output = renderer.render(frame, ("Input: ISOLATED_WORD", "Word: cat", "State: READY"))

    assert output.shape == frame.shape
    assert np.array_equal(frame, np.zeros_like(frame))
    assert np.array_equal(output[:, :320], frame[:, :320])
    assert np.any(output[:120, 320:] != 0)
    assert np.array_equal(output[180:, :], frame[180:, :])


def test_hud_wraps_long_lines_to_available_width() -> None:
    renderer = StatusHudRenderer(target_font_height_px=13)

    wrapped = renderer._wrap_lines(
        ("Prediction review contains several candidates that need more horizontal space",),
        150,
    )

    assert len(wrapped) > 1
    assert all(renderer._text_size(line)[0] <= 150 for line in wrapped)


def test_empty_hud_returns_an_unmodified_copy() -> None:
    renderer = StatusHudRenderer()
    frame = np.full((80, 120, 3), 50, dtype=np.uint8)

    output = renderer.render(frame, ())

    assert output is not frame
    assert np.array_equal(output, frame)
