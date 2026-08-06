import numpy as np
from numpy.typing import NDArray

from app.ml.exceptions import InvalidEMNISTLabelError


class EMNISTLabelMapper:
    RAW_LABEL_MIN = 1
    RAW_LABEL_MAX = 26

    @classmethod
    def map_raw_label(cls, raw_label: int) -> int:
        if isinstance(raw_label, bool) or not isinstance(raw_label, (int, np.integer)):
            raise InvalidEMNISTLabelError("Raw EMNIST label must be an integer from 1 to 26")
        value = int(raw_label)
        if value < cls.RAW_LABEL_MIN or value > cls.RAW_LABEL_MAX:
            raise InvalidEMNISTLabelError(f"Raw EMNIST label must be 1-26, got {value}")
        return value - 1

    @classmethod
    def map_raw_labels(cls, raw_labels: object) -> NDArray[np.int64]:
        if not isinstance(raw_labels, np.ndarray):
            raise InvalidEMNISTLabelError("Raw EMNIST labels must be a NumPy array")
        if raw_labels.ndim != 1 or raw_labels.size == 0:
            raise InvalidEMNISTLabelError("Raw EMNIST labels must be a non-empty 1D array")
        if not np.issubdtype(raw_labels.dtype, np.integer):
            raise InvalidEMNISTLabelError("Raw EMNIST labels must use an integer dtype")
        minimum = int(raw_labels.min())
        maximum = int(raw_labels.max())
        if minimum < cls.RAW_LABEL_MIN or maximum > cls.RAW_LABEL_MAX:
            raise InvalidEMNISTLabelError(
                f"Raw EMNIST labels must stay in 1-26, observed {minimum}-{maximum}"
            )
        return raw_labels.astype(np.int64, copy=True) - 1
