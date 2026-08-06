from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInputContract:
    width: int = 28
    height: int = 28
    channels: int = 1
    background_value: int = 0
    normalization_divisor: float = 255.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Model input width and height must be greater than 0")
        if self.channels != 1:
            raise ValueError("Model input channels must be 1 for grayscale images")
        if not 0 <= self.background_value <= 255:
            raise ValueError("Model input background value must be between 0 and 255")
        if self.normalization_divisor <= 0.0:
            raise ValueError("Model input normalization divisor must be greater than 0")

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def model_sample_shape(self) -> tuple[int, int, int]:
        return self.height, self.width, self.channels
