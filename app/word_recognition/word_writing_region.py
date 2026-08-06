from dataclasses import dataclass


@dataclass(frozen=True)
class WordWritingRegion:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("Word writing region coordinates must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Word writing region dimensions must be positive")

    @classmethod
    def from_ratios(
        cls,
        canvas_width: int,
        canvas_height: int,
        *,
        x_ratio: float,
        y_ratio: float,
        width_ratio: float,
        height_ratio: float,
    ) -> "WordWritingRegion":
        if canvas_width <= 0 or canvas_height <= 0:
            raise ValueError("Canvas dimensions must be positive")
        ratios = (x_ratio, y_ratio, width_ratio, height_ratio)
        if any(value < 0.0 or value > 1.0 for value in ratios):
            raise ValueError("Word ROI ratios must be between 0 and 1")
        if width_ratio <= 0.0 or height_ratio <= 0.0:
            raise ValueError("Word ROI size ratios must be positive")
        if x_ratio + width_ratio > 1.0 or y_ratio + height_ratio > 1.0:
            raise ValueError("Word ROI must fit inside the canvas")
        x = int(round(canvas_width * x_ratio))
        y = int(round(canvas_height * y_ratio))
        width = min(int(round(canvas_width * width_ratio)), canvas_width - x)
        height = min(int(round(canvas_height * height_ratio)), canvas_height - y)
        return cls(x=x, y=y, width=width, height=height)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def contains(self, point: tuple[int, int]) -> bool:
        x, y = point
        return self.x <= x < self.right and self.y <= y < self.bottom
