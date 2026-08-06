from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from app.preprocessing.bounding_box import BoundingBox
from app.word_recognition.character_segment import CharacterSegment
from app.word_recognition.connected_component_extractor import ConnectedComponentExtractor
from app.word_recognition.foreground_component import ForegroundComponent
from app.word_recognition.segmentation_result import WordSegmentationResult
from app.word_recognition.segmentation_status import SegmentationStatus
from app.word_recognition.segmentation_validator import SegmentationValidator
from app.word_recognition.stroke_group import StrokeGroup
from app.word_recognition.stroke_group_builder import StrokeGroupBuilder
from app.word_recognition.vertical_projection_analyzer import VerticalProjectionAnalyzer
from app.word_recognition.word_input_snapshot import WordInputSnapshot


class HybridWordSegmenter:
    def __init__(
        self,
        *,
        min_foreground_pixels: int = 10,
        min_component_area: int = 4,
        min_separator_gap: int = 10,
        min_characters: int = 2,
        max_characters: int = 12,
        wide_group_ratio: float = 1.80,
        binary_threshold: int = 20,
        x_overlap_threshold: float = 0.25,
        tiny_component_ratio: float = 0.20,
        max_internal_gap: int = 16,
    ) -> None:
        if min_foreground_pixels <= 0 or not 0 <= binary_threshold <= 255:
            raise ValueError("Foreground settings are invalid")
        if wide_group_ratio <= 1.0:
            raise ValueError("wide_group_ratio must be greater than 1")
        self.min_foreground_pixels = min_foreground_pixels
        self.binary_threshold = binary_threshold
        self.wide_group_ratio = wide_group_ratio
        if max_internal_gap <= 0:
            raise ValueError("max_internal_gap must be positive")
        self.max_internal_gap = max_internal_gap
        self.component_extractor = ConnectedComponentExtractor(min_component_area)
        self.projection_analyzer = VerticalProjectionAnalyzer(min_separator_gap)
        self.stroke_group_builder = StrokeGroupBuilder(
            x_overlap_threshold,
            tiny_component_ratio,
        )
        self.validator = SegmentationValidator(min_characters, max_characters)

    def segment(self, snapshot: WordInputSnapshot) -> WordSegmentationResult:
        started_at = perf_counter()
        region = snapshot.writing_region
        roi = snapshot.canvas_image[region.y : region.bottom, region.x : region.right]
        grayscale = self._to_grayscale(roi)
        binary = np.where(grayscale > self.binary_threshold, 255, 0).astype(np.uint8)
        components = self.component_extractor.extract(binary)
        cleaned = np.zeros_like(binary)
        for component in components:
            for y, x in component.pixels:
                cleaned[y, x] = 255
        groups = self.stroke_group_builder.build(components)
        if int(np.count_nonzero(cleaned)) < self.min_foreground_pixels:
            return self._result(
                SegmentationStatus.EMPTY,
                (),
                None,
                cleaned,
                components,
                "No whole-word foreground found",
                started_at,
            )

        ys, xs = np.where(cleaned > 0)
        word_box_local = BoundingBox(
            x=int(xs.min()),
            y=int(ys.min()),
            width=int(xs.max() - xs.min() + 1),
            height=int(ys.max() - ys.min() + 1),
        )
        word_binary = cleaned[
            word_box_local.y : word_box_local.bottom,
            word_box_local.x : word_box_local.right,
        ]
        word_grayscale = grayscale[
            word_box_local.y : word_box_local.bottom,
            word_box_local.x : word_box_local.right,
        ]
        ranges = self._character_ranges(
            word_binary,
            groups,
            word_box_local,
            region.x,
            snapshot,
        )
        segments = tuple(
            self._make_segment(
                position,
                start,
                end,
                word_binary,
                word_grayscale,
                word_box_local,
                region.x,
                region.y,
                snapshot,
            )
            for position, (start, end) in enumerate(ranges)
        )
        status = self.validator.validate(segments)
        if status is SegmentationStatus.SUCCESS and self._contains_suspicious_wide_segment(
            segments
        ):
            status = SegmentationStatus.AMBIGUOUS
        global_box = BoundingBox(
            x=region.x + word_box_local.x,
            y=region.y + word_box_local.y,
            width=word_box_local.width,
            height=word_box_local.height,
        )
        return self._result(
            status,
            segments,
            global_box,
            cleaned,
            components,
            f"Whole-word segmentation: {status.value.lower()}",
            started_at,
        )

    def split_segment(self, segment: CharacterSegment) -> tuple[CharacterSegment, ...] | None:
        binary = np.where(segment.grayscale_image > self.binary_threshold, 255, 0).astype(np.uint8)
        split_x = self.projection_analyzer.best_internal_split(binary)
        if split_x is None or split_x <= 0 or split_x >= binary.shape[1] - 1:
            return None
        children: list[CharacterSegment] = []
        for index, (start, end) in enumerate(((0, split_x), (split_x, binary.shape[1]))):
            crop = segment.grayscale_image[:, start:end]
            if not np.any(crop > self.binary_threshold):
                return None
            children.append(
                CharacterSegment(
                    segment_id=f"{segment.segment_id}_split_{index}",
                    position=segment.position + index,
                    bounding_box=BoundingBox(
                        segment.bounding_box.x + start,
                        segment.bounding_box.y,
                        end - start,
                        segment.bounding_box.height,
                    ),
                    source_stroke_ids=segment.source_stroke_ids,
                    grayscale_image=crop,
                    segmentation_confidence=0.55,
                )
            )
        return tuple(children)

    @staticmethod
    def merge_segments(
        first: CharacterSegment,
        second: CharacterSegment,
    ) -> CharacterSegment:
        left = min(first.bounding_box.x, second.bounding_box.x)
        top = min(first.bounding_box.y, second.bounding_box.y)
        right = max(first.bounding_box.right, second.bounding_box.right)
        bottom = max(first.bounding_box.bottom, second.bounding_box.bottom)
        image = np.zeros((bottom - top, right - left), dtype=np.uint8)
        for segment in (first, second):
            x = segment.bounding_box.x - left
            y = segment.bounding_box.y - top
            height, width = segment.grayscale_image.shape
            target = image[y : y + height, x : x + width]
            np.maximum(target, segment.grayscale_image, out=target)
        return CharacterSegment(
            segment_id=f"{first.segment_id}_merge_{second.segment_id}",
            position=first.position,
            bounding_box=BoundingBox(left, top, right - left, bottom - top),
            source_stroke_ids=tuple(
                dict.fromkeys(first.source_stroke_ids + second.source_stroke_ids)
            ),
            grayscale_image=image,
            segmentation_confidence=min(
                first.segmentation_confidence,
                second.segmentation_confidence,
                0.6,
            ),
        )

    def _character_ranges(
        self,
        binary: NDArray[np.uint8],
        groups: tuple[StrokeGroup, ...],
        word_box_local: BoundingBox,
        roi_x: int,
        snapshot: WordInputSnapshot,
    ) -> tuple[tuple[int, int], ...]:
        projection = self.projection_analyzer.projection(binary)
        occupied = np.flatnonzero(projection > 0)
        if occupied.size == 0:
            return ()
        gaps = self.projection_analyzer.find_gaps(binary)
        internal = []
        for gap in gaps:
            if gap.start_x <= int(occupied.min()) or gap.end_x > int(occupied.max()):
                continue
            roi_midpoint = word_box_local.x + (gap.start_x + gap.end_x) / 2.0
            global_midpoint = roi_x + roi_midpoint
            inside_detached_group = any(
                len(group.component_ids) > 1
                and group.bounding_box.x < roi_midpoint < group.bounding_box.right
                for group in groups
            )
            crossed_by_stroke = (
                any(
                    stroke.bounding_box.x < global_midpoint < stroke.bounding_box.right
                    for stroke in snapshot.strokes
                )
                and gap.width <= self.max_internal_gap
            )
            if not inside_detached_group and not crossed_by_stroke:
                internal.append(gap)
        ranges: list[tuple[int, int]] = []
        start = int(occupied.min())
        for gap in internal:
            ranges.append((start, gap.start_x))
            start = gap.end_x
        ranges.append((start, int(occupied.max()) + 1))
        return tuple((start, end) for start, end in ranges if end > start)

    def _make_segment(
        self,
        position: int,
        start: int,
        end: int,
        word_binary: NDArray[np.uint8],
        word_grayscale: NDArray[np.uint8],
        word_box_local: BoundingBox,
        roi_x: int,
        roi_y: int,
        snapshot: WordInputSnapshot,
    ) -> CharacterSegment:
        crop_binary = word_binary[:, start:end]
        ys, xs = np.where(crop_binary > 0)
        x_start = start + int(xs.min())
        x_end = start + int(xs.max()) + 1
        y_start = int(ys.min())
        y_end = int(ys.max()) + 1
        crop = word_grayscale[y_start:y_end, x_start:x_end]
        box = BoundingBox(
            x=roi_x + word_box_local.x + x_start,
            y=roi_y + word_box_local.y + y_start,
            width=x_end - x_start,
            height=y_end - y_start,
        )
        stroke_ids = tuple(
            stroke.stroke_id
            for stroke in snapshot.strokes
            if self._boxes_intersect(box, stroke.bounding_box)
        )
        return CharacterSegment(
            segment_id=f"{snapshot.snapshot_id}:segment:{position}",
            position=position,
            bounding_box=box,
            source_stroke_ids=stroke_ids,
            grayscale_image=crop,
            segmentation_confidence=0.9,
        )

    def _contains_suspicious_wide_segment(
        self,
        segments: tuple[CharacterSegment, ...],
    ) -> bool:
        if len(segments) < 2:
            return False
        widths = sorted(segment.bounding_box.width for segment in segments)
        median = widths[len(widths) // 2]
        return any(
            segment.bounding_box.width / max(1, median) > self.wide_group_ratio
            for segment in segments
        )

    @staticmethod
    def _boxes_intersect(first: BoundingBox, second: BoundingBox) -> bool:
        return not (
            first.right <= second.x
            or second.right <= first.x
            or first.bottom <= second.y
            or second.bottom <= first.y
        )

    @staticmethod
    def _to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if image.ndim == 2:
            return image.copy()
        if image.ndim != 3 or image.shape[2] not in {3, 4}:
            raise ValueError("Word canvas must be grayscale, BGR, or BGRA")
        return image[..., :3].max(axis=2).astype(np.uint8)

    @staticmethod
    def _result(
        status: SegmentationStatus,
        segments: tuple[CharacterSegment, ...],
        word_box: BoundingBox | None,
        foreground_mask: NDArray[np.uint8],
        components: tuple[ForegroundComponent, ...],
        message: str,
        started_at: float,
    ) -> WordSegmentationResult:
        return WordSegmentationResult(
            status=status,
            segments=segments,
            word_bounding_box=word_box,
            foreground_mask=foreground_mask,
            components=components,
            message=message,
            elapsed_ms=(perf_counter() - started_at) * 1000.0,
        )
