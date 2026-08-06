from collections import deque

import numpy as np
from numpy.typing import NDArray

from app.preprocessing.bounding_box import BoundingBox
from app.word_recognition.foreground_component import ForegroundComponent


class ConnectedComponentExtractor:
    def __init__(self, min_component_area: int = 4) -> None:
        if min_component_area <= 0:
            raise ValueError("min_component_area must be positive")
        self.min_component_area = min_component_area

    def extract(self, binary: NDArray[np.uint8]) -> tuple[ForegroundComponent, ...]:
        if not isinstance(binary, np.ndarray) or binary.dtype != np.uint8 or binary.ndim != 2:
            raise ValueError("binary must be a 2D uint8 array")
        foreground = binary > 0
        visited = np.zeros(foreground.shape, dtype=bool)
        components: list[ForegroundComponent] = []
        height, width = foreground.shape
        for y in range(height):
            for x in range(width):
                if not foreground[y, x] or visited[y, x]:
                    continue
                pixels = self._flood_fill(foreground, visited, x, y)
                if len(pixels) < self.min_component_area:
                    continue
                xs = [point[1] for point in pixels]
                ys = [point[0] for point in pixels]
                components.append(
                    ForegroundComponent(
                        component_id=f"component_{len(components):03d}",
                        bounding_box=BoundingBox(
                            x=min(xs),
                            y=min(ys),
                            width=max(xs) - min(xs) + 1,
                            height=max(ys) - min(ys) + 1,
                        ),
                        area=len(pixels),
                        centroid_x=float(sum(xs) / len(xs)),
                        centroid_y=float(sum(ys) / len(ys)),
                        pixels=tuple(pixels),
                    )
                )
        return tuple(components)

    @staticmethod
    def _flood_fill(
        foreground: NDArray[np.bool_],
        visited: NDArray[np.bool_],
        start_x: int,
        start_y: int,
    ) -> list[tuple[int, int]]:
        height, width = foreground.shape
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if delta_x == 0 and delta_y == 0:
                        continue
                    neighbour_x = x + delta_x
                    neighbour_y = y + delta_y
                    if not (0 <= neighbour_x < width and 0 <= neighbour_y < height):
                        continue
                    if (
                        foreground[neighbour_y, neighbour_x]
                        and not visited[neighbour_y, neighbour_x]
                    ):
                        visited[neighbour_y, neighbour_x] = True
                        queue.append((neighbour_y, neighbour_x))
        return pixels
