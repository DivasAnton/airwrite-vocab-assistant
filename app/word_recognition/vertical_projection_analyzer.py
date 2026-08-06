from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ProjectionGap:
    start_x: int
    end_x: int
    width: int
    score: float


class VerticalProjectionAnalyzer:
    def __init__(self, min_separator_gap: int = 10) -> None:
        if min_separator_gap <= 0:
            raise ValueError("min_separator_gap must be positive")
        self.min_separator_gap = min_separator_gap

    def projection(self, binary: NDArray[np.uint8]) -> NDArray[np.int64]:
        if binary.ndim != 2:
            raise ValueError("binary image must be 2D")
        return np.count_nonzero(binary, axis=0).astype(np.int64)

    def find_gaps(self, binary: NDArray[np.uint8]) -> tuple[ProjectionGap, ...]:
        projection = self.projection(binary)
        gaps: list[ProjectionGap] = []
        start: int | None = None
        for index, value in enumerate(projection):
            if value == 0 and start is None:
                start = index
            if value > 0 and start is not None:
                self._append_gap(gaps, start, index)
                start = None
        if start is not None:
            self._append_gap(gaps, start, len(projection))
        return tuple(gaps)

    def best_internal_split(self, binary: NDArray[np.uint8]) -> int | None:
        projection = self.projection(binary)
        if len(projection) < 3:
            return None
        margin = max(1, len(projection) // 6)
        candidates = projection[margin:-margin]
        if candidates.size == 0:
            return None
        minimum = int(candidates.min())
        maximum = int(projection.max())
        if maximum == 0 or minimum > maximum * 0.35:
            return None
        local_index = int(np.argmin(candidates))
        return margin + local_index

    def _append_gap(self, gaps: list[ProjectionGap], start: int, end: int) -> None:
        width = end - start
        if width >= self.min_separator_gap:
            gaps.append(
                ProjectionGap(
                    start_x=start,
                    end_x=end,
                    width=width,
                    score=min(1.0, width / (self.min_separator_gap * 2.0)),
                )
            )
