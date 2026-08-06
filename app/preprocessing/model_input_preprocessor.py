from typing import cast

import numpy as np
from numpy.typing import NDArray

from app.preprocessing.bounding_box import BoundingBox
from app.preprocessing.exceptions import InvalidImageError
from app.preprocessing.model_input_contract import ModelInputContract
from app.preprocessing.preprocessing_result import PreprocessingResult
from app.preprocessing.preprocessing_source import PreprocessingSource


class ModelInputPreprocessor:
    def __init__(
        self,
        contract: ModelInputContract | None = None,
        validate_dark_background: bool = True,
    ) -> None:
        self.contract = contract or ModelInputContract()
        self.validate_dark_background = validate_dark_background

    def process(
        self,
        prepared_image: object,
        *,
        source: PreprocessingSource,
        orientation_transform: str = "none",
        original_shape: tuple[int, ...] | None = None,
        bounding_box: BoundingBox | None = None,
        cropped_shape: tuple[int, int] | None = None,
        foreground_pixel_count: int | None = None,
        debug_images: dict[str, NDArray[np.uint8]] | None = None,
    ) -> PreprocessingResult:
        image = self._validate_prepared_image(prepared_image)
        if self.validate_dark_background:
            self._validate_polarity(image)

        processed = image.copy()
        normalized = cast(
            NDArray[np.float32],
            processed.astype(np.float32) / self.contract.normalization_divisor,
        )
        if not np.isfinite(normalized).all():
            raise InvalidImageError("Normalized model input contains NaN or infinite values")
        if normalized.size and (normalized.min() < 0.0 or normalized.max() > 1.0):
            raise InvalidImageError("Normalized model input must stay in range 0.0 to 1.0")

        mask = processed > self.contract.background_value
        model_foreground_count = int(np.count_nonzero(mask))
        count = model_foreground_count if foreground_pixel_count is None else foreground_pixel_count
        model_bbox = bounding_box or self._find_model_bbox(mask)
        shape = original_shape or tuple(processed.shape)
        crop_shape = cropped_shape or tuple(processed.shape)
        return PreprocessingResult(
            processed_image=processed,
            normalized_image=normalized,
            bounding_box=model_bbox,
            original_shape=shape,
            cropped_shape=crop_shape,
            foreground_pixel_count=count,
            debug_images={} if debug_images is None else dict(debug_images),
            source=source,
            orientation_transform=orientation_transform,
            foreground_ratio=model_foreground_count / processed.size,
        )

    def normalize_batch(self, prepared_images: object) -> NDArray[np.float32]:
        if not isinstance(prepared_images, np.ndarray):
            raise InvalidImageError("Prepared model input batch must be a NumPy array")
        if prepared_images.dtype != np.uint8:
            raise InvalidImageError("Prepared model input batch must use uint8 dtype")
        if prepared_images.ndim != 3 or prepared_images.shape[1:] != self.contract.image_shape:
            raise InvalidImageError(
                f"Prepared model input batch shape must be (N,{self.contract.height},"
                f"{self.contract.width}), got {prepared_images.shape}"
            )
        if len(prepared_images) == 0:
            raise InvalidImageError("Prepared model input batch must not be empty")
        normalized = prepared_images.astype(np.float32) / self.contract.normalization_divisor
        if not np.isfinite(normalized).all() or normalized.min() < 0.0 or normalized.max() > 1.0:
            raise InvalidImageError(
                "Normalized model input batch must be finite and stay in 0.0-1.0"
            )
        return cast(NDArray[np.float32], normalized)

    def _validate_prepared_image(self, image: object) -> NDArray[np.uint8]:
        if not isinstance(image, np.ndarray):
            raise InvalidImageError("Prepared model input must be a NumPy array")
        if image.dtype != np.uint8:
            raise InvalidImageError("Prepared model input must use uint8 dtype")
        if image.ndim != 2 or image.shape != self.contract.image_shape:
            raise InvalidImageError(
                f"Prepared model input shape must be {self.contract.image_shape}, got {image.shape}"
            )
        return cast(NDArray[np.uint8], image)

    def _validate_polarity(self, image: NDArray[np.uint8]) -> None:
        corners = np.array(
            [image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]],
            dtype=np.float32,
        )
        if float(corners.mean()) > 127.5:
            raise InvalidImageError(
                "Model input appears to have a light background; expected dark background"
            )

    @staticmethod
    def _find_model_bbox(mask: NDArray[np.bool_]) -> BoundingBox | None:
        if not np.any(mask):
            return None
        ys, xs = np.where(mask)
        x = int(xs.min())
        y = int(ys.min())
        return BoundingBox(
            x=x,
            y=y,
            width=int(xs.max()) - x + 1,
            height=int(ys.max()) - y + 1,
        )
