from app.word_recognition.foreground_component import ForegroundComponent
from app.word_recognition.stroke_group import StrokeGroup
from app.word_recognition.stroke_group_builder import StrokeGroupBuilder


class DetachedComponentMerger:
    def __init__(
        self, tiny_component_ratio: float = 0.20, x_overlap_threshold: float = 0.25
    ) -> None:
        if not 0.0 < tiny_component_ratio <= 1.0:
            raise ValueError("tiny_component_ratio must be between 0 and 1")
        self.tiny_component_ratio = tiny_component_ratio
        self._builder = StrokeGroupBuilder(x_overlap_threshold, tiny_component_ratio)

    def merge(
        self,
        components: tuple[ForegroundComponent, ...],
    ) -> tuple[StrokeGroup, ...]:
        return self._builder.build(components)
