class DatasetError(Exception):
    """Base exception for dataset preparation failures."""


class DatasetStructureError(DatasetError):
    """Raised when the dataset directory does not satisfy the A-Z contract."""


class DatasetSplitError(DatasetError):
    """Raised when valid samples cannot be split without leakage."""
