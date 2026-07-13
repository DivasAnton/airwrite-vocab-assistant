from app.drawing.stroke_manager import LineSegment, StrokeManager


def test_first_point_does_not_create_segment() -> None:
    manager = StrokeManager()

    assert manager.update((100, 100)) is None


def test_second_point_creates_segment() -> None:
    manager = StrokeManager()
    manager.update((100, 100))

    segment = manager.update((110, 105))

    assert segment == LineSegment(start=(100, 100), end=(110, 105))


def test_lost_tracking_resets_previous_point() -> None:
    manager = StrokeManager()
    manager.update((100, 100))
    manager.update(None)

    segment = manager.update((500, 400))

    assert segment is None


def test_large_jump_does_not_create_segment() -> None:
    manager = StrokeManager(max_point_distance=50)
    manager.update((100, 100))

    segment = manager.update((900, 500))

    assert segment is None


def test_reset_clears_previous_point() -> None:
    manager = StrokeManager()
    manager.update((100, 100))
    manager.reset()

    assert manager.update((120, 120)) is None
