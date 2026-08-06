import numpy as np

from app.word_recognition.connected_component_extractor import ConnectedComponentExtractor
from app.word_recognition.hybrid_word_segmenter import HybridWordSegmenter
from app.word_recognition.segmentation_status import SegmentationStatus
from app.word_recognition.vertical_projection_analyzer import VerticalProjectionAnalyzer
from app.word_recognition.word_input_snapshot import WordInputSnapshot
from app.word_recognition.word_writing_region import WordWritingRegion


def word_snapshot(*ranges: tuple[int, int]) -> WordInputSnapshot:
    canvas = np.zeros((100, 240, 3), dtype=np.uint8)
    for left, right in ranges:
        canvas[25:75, left:right] = 255
    return WordInputSnapshot(
        snapshot_id="word-1",
        canvas_image=canvas,
        strokes=(),
        writing_region=WordWritingRegion(0, 0, 240, 100),
        completed_at_ms=1,
    )


def test_connected_components_use_eight_connectivity_and_filter_noise() -> None:
    image = np.zeros((12, 12), dtype=np.uint8)
    image[2:5, 2:5] = 255
    image[5, 5] = 255
    image[10, 10] = 255

    components = ConnectedComponentExtractor(min_component_area=2).extract(image)

    assert len(components) == 1
    assert components[0].area == 10


def test_projection_finds_visible_inter_character_gap() -> None:
    image = np.zeros((20, 60), dtype=np.uint8)
    image[:, 3:18] = 255
    image[:, 35:52] = 255

    gaps = VerticalProjectionAnalyzer(min_separator_gap=10).find_gaps(image)

    assert any(gap.start_x == 18 and gap.end_x == 35 for gap in gaps)


def test_hybrid_segmenter_returns_left_to_right_isolated_letters() -> None:
    result = HybridWordSegmenter(min_separator_gap=8).segment(
        word_snapshot((15, 45), (65, 95), (120, 150))
    )

    assert result.status is SegmentationStatus.SUCCESS
    assert len(result.segments) == 3
    assert [segment.position for segment in result.segments] == [0, 1, 2]
    assert [segment.bounding_box.x for segment in result.segments] == [15, 65, 120]


def test_hybrid_segmenter_rejects_empty_and_too_few_letters() -> None:
    segmenter = HybridWordSegmenter(min_separator_gap=8)
    empty = np.zeros((100, 240, 3), dtype=np.uint8)
    empty_result = segmenter.segment(
        WordInputSnapshot(
            "empty",
            empty,
            (),
            WordWritingRegion(0, 0, 240, 100),
            1,
        )
    )

    assert empty_result.status is SegmentationStatus.EMPTY
    assert segmenter.segment(word_snapshot((15, 45))).status is SegmentationStatus.TOO_FEW_SEGMENTS


def test_split_and_merge_are_explicit_corrections() -> None:
    segmenter = HybridWordSegmenter(min_separator_gap=8)
    result = segmenter.segment(word_snapshot((10, 30), (45, 65), (90, 110)))

    merged = segmenter.merge_segments(result.segments[0], result.segments[1])
    split = segmenter.split_segment(merged)

    assert merged.bounding_box.x == 10
    assert split is not None
    assert len(split) == 2
