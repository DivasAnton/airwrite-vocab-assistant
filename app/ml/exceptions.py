class DatasetError(Exception):
    """Base exception for dataset preparation failures."""


class DatasetStructureError(DatasetError):
    """Raised when the dataset directory does not satisfy the A-Z contract."""


class DatasetSplitError(DatasetError):
    """Raised when valid samples cannot be split without leakage."""


class DatasetLoadError(DatasetError):
    """Raised when a manifest image cannot be loaded for training."""


class TrainingDependencyError(RuntimeError):
    """Raised when an optional model training dependency is unavailable."""


class ModelEvaluationError(RuntimeError):
    """Raised when model outputs do not satisfy the evaluation contract."""


class ArtifactExportError(RuntimeError):
    """Raised when model metadata or artifacts cannot be exported safely."""
