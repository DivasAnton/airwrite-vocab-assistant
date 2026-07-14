from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError(
                f"BoundingBox coordinates must be non-negative, got {self.x}, {self.y}"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"BoundingBox width and height must be positive, got {self.width}x{self.height}"
            )

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0
