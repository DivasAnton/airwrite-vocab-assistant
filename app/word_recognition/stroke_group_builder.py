from app.preprocessing.bounding_box import BoundingBox
from app.word_recognition.foreground_component import ForegroundComponent
from app.word_recognition.recorded_stroke import RecordedStroke
from app.word_recognition.stroke_group import StrokeGroup


class StrokeGroupBuilder:
    def __init__(
        self,
        x_overlap_threshold: float = 0.25,
        tiny_component_ratio: float = 0.20,
    ) -> None:
        if not 0.0 <= x_overlap_threshold <= 1.0:
            raise ValueError("x_overlap_threshold must be between 0 and 1")
        if not 0.0 < tiny_component_ratio <= 1.0:
            raise ValueError("tiny_component_ratio must be between 0 and 1")
        self.x_overlap_threshold = x_overlap_threshold
        self.tiny_component_ratio = tiny_component_ratio

    def build(
        self,
        components: tuple[ForegroundComponent, ...],
        strokes: tuple[RecordedStroke, ...] = (),
    ) -> tuple[StrokeGroup, ...]:
        if not components:
            return ()
        parents = list(range(len(components)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parents[second_root] = first_root

        median_area = sorted(component.area for component in components)[len(components) // 2]
        for first in range(len(components)):
            for second in range(first + 1, len(components)):
                if self._should_merge(components[first], components[second], median_area):
                    union(first, second)

        grouped: dict[int, list[ForegroundComponent]] = {}
        for index, component in enumerate(components):
            grouped.setdefault(find(index), []).append(component)

        groups: list[StrokeGroup] = []
        for members in grouped.values():
            bounding_box = self._union_boxes(tuple(member.bounding_box for member in members))
            stroke_ids = tuple(
                stroke.stroke_id
                for stroke in strokes
                if self._boxes_intersect(bounding_box, stroke.bounding_box)
            )
            groups.append(
                StrokeGroup(
                    group_id=f"group_{len(groups):03d}",
                    bounding_box=bounding_box,
                    component_ids=tuple(member.component_id for member in members),
                    source_stroke_ids=stroke_ids,
                    grouping_confidence=1.0 if len(members) == 1 else 0.8,
                )
            )
        return tuple(sorted(groups, key=lambda group: group.bounding_box.center[0]))

    def _should_merge(
        self,
        first: ForegroundComponent,
        second: ForegroundComponent,
        median_area: int,
    ) -> bool:
        overlap = self._x_overlap_ratio(first.bounding_box, second.bounding_box)
        smaller, larger = sorted((first, second), key=lambda component: component.area)
        tiny = smaller.area <= max(4, int(median_area * self.tiny_component_ratio))
        vertical_gap = max(
            larger.bounding_box.y - smaller.bounding_box.bottom,
            smaller.bounding_box.y - larger.bounding_box.bottom,
            0,
        )
        dot_body = (
            tiny
            and overlap >= self.x_overlap_threshold
            and vertical_gap <= larger.bounding_box.height
        )
        crossbar = (
            smaller.bounding_box.width > smaller.bounding_box.height * 2
            and overlap >= self.x_overlap_threshold
            and vertical_gap <= larger.bounding_box.height // 2 + 2
        )
        return dot_body or crossbar

    @staticmethod
    def _x_overlap_ratio(first: BoundingBox, second: BoundingBox) -> float:
        overlap = max(0, min(first.right, second.right) - max(first.x, second.x))
        return overlap / max(1, min(first.width, second.width))

    @staticmethod
    def _boxes_intersect(first: BoundingBox, second: BoundingBox) -> bool:
        return not (
            first.right <= second.x
            or second.right <= first.x
            or first.bottom <= second.y
            or second.bottom <= first.y
        )

    @staticmethod
    def _union_boxes(boxes: tuple[BoundingBox, ...]) -> BoundingBox:
        left = min(box.x for box in boxes)
        top = min(box.y for box in boxes)
        right = max(box.right for box in boxes)
        bottom = max(box.bottom for box in boxes)
        return BoundingBox(left, top, right - left, bottom - top)
