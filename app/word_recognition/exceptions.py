class WholeWordRecognitionError(RuntimeError):
    """Base error for whole-word recognition failures."""


class InvalidWordInputError(WholeWordRecognitionError):
    """Raised when a whole-word snapshot or batch violates its contract."""


class SegmentationError(WholeWordRecognitionError):
    """Raised when character segmentation cannot produce a valid result."""


class DraftOperationError(WholeWordRecognitionError):
    """Raised when an invalid draft correction is requested."""
