class InferenceError(Exception):
    """Base exception for character inference."""


class ModelArtifactNotFoundError(InferenceError):
    """Raised when a required model artifact is missing."""


class InvalidModelBundleError(InferenceError):
    """Raised when model artifacts are invalid or incompatible."""


class InvalidModelInputError(InferenceError):
    """Raised when a preprocessed model input is invalid."""


class CharacterPredictionError(InferenceError):
    """Raised when model inference fails or returns invalid output."""
