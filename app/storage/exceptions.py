class DrawingStorageError(Exception):
    """Base exception for drawing storage failures."""


class InvalidDrawingImageError(DrawingStorageError):
    """Raised when the drawing image is invalid."""


class DrawingImageSaveError(DrawingStorageError):
    """Raised when a drawing image cannot be written."""
