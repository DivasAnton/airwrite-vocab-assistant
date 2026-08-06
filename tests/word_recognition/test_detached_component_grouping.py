import numpy as np

from app.word_recognition.connected_component_extractor import ConnectedComponentExtractor
from app.word_recognition.stroke_group_builder import StrokeGroupBuilder


def test_dot_and_body_are_grouped_as_one_character_hint() -> None:
    image = np.zeros((50, 30), dtype=np.uint8)
    image[5:9, 13:17] = 255
    image[16:44, 12:18] = 255
    components = ConnectedComponentExtractor(min_component_area=4).extract(image)

    groups = StrokeGroupBuilder(x_overlap_threshold=0.25).build(components)

    assert len(components) == 2
    assert len(groups) == 1
    assert len(groups[0].component_ids) == 2


def test_crossbar_and_body_are_grouped_as_one_character_hint() -> None:
    image = np.zeros((50, 40), dtype=np.uint8)
    image[20:45, 18:22] = 255
    image[14:17, 8:32] = 255
    components = ConnectedComponentExtractor(min_component_area=4).extract(image)

    groups = StrokeGroupBuilder(x_overlap_threshold=0.25).build(components)

    assert len(components) == 2
    assert len(groups) == 1
