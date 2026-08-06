from typing import cast

import numpy as np
from numpy.typing import NDArray

from app.preprocessing.exceptions import InvalidImageError
from app.preprocessing.model_input_contract import ModelInputContract


class EMNISTSourceAdapter:
    def __init__(
        self,
        contract: ModelInputContract | None = None,
        transpose_images: bool = True,
        preserve_grayscale: bool = True,
    ) -> None:
        if not preserve_grayscale:
            raise ValueError("EMNIST grayscale preservation must remain enabled in Sprint 8E")
        self.contract = contract or ModelInputContract()
        self.transpose_images = transpose_images
        self.preserve_grayscale = preserve_grayscale

    @property
    def orientation_transform(self) -> str:
        return "transpose" if self.transpose_images else "none"

    def adapt(self, image: object) -> NDArray[np.uint8]:
        try:
            array = np.asarray(image)
        except Exception as error:
            raise InvalidImageError("EMNIST input must be convertible to a NumPy array") from error

        expected_2d = self.contract.image_shape
        expected_3d = (*expected_2d, 1)
        if array.shape == expected_3d:
            array = np.squeeze(array, axis=2)
        elif array.shape != expected_2d:
            raise InvalidImageError(
                f"EMNIST input shape must be {expected_2d} or {expected_3d}, got {array.shape}"
            )
        if array.dtype != np.uint8:
            raise InvalidImageError("EMNIST input must use uint8 dtype")

        prepared = np.transpose(array) if self.transpose_images else array.copy()
        return cast(NDArray[np.uint8], np.ascontiguousarray(prepared, dtype=np.uint8))

    def adapt_batch(self, images: object) -> NDArray[np.uint8]:
        if not isinstance(images, np.ndarray):
            raise InvalidImageError("EMNIST image batch must be a NumPy array")
        expected_3d = (self.contract.height, self.contract.width)
        if images.ndim == 4 and images.shape[1:] == (*expected_3d, 1):
            batch = np.squeeze(images, axis=3)
        elif images.ndim == 3 and images.shape[1:] == expected_3d:
            batch = images
        else:
            raise InvalidImageError(
                f"EMNIST image batch shape must be (N,28,28) or (N,28,28,1), got {images.shape}"
            )
        if batch.dtype != np.uint8 or len(batch) == 0:
            raise InvalidImageError("EMNIST image batch must be non-empty and use uint8 dtype")
        prepared = np.transpose(batch, (0, 2, 1)) if self.transpose_images else batch.copy()
        return cast(NDArray[np.uint8], np.ascontiguousarray(prepared, dtype=np.uint8))
