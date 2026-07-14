class PreprocessingError(Exception):
    """Base exception for handwriting preprocessing."""


class InvalidImageError(PreprocessingError):
    """Raised when an input image is invalid."""


class EmptyDrawingError(PreprocessingError):
    """Raised when no meaningful drawing is found."""


class InvalidPreprocessingConfigError(PreprocessingError):
    """Raised when preprocessing configuration is invalid."""
