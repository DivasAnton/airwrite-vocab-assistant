class WordBuilderError(Exception):
    """Base exception for word-building operations."""


class InvalidCharacterError(WordBuilderError):
    """Raised when a character is not supported."""


class WordLengthLimitError(WordBuilderError):
    """Raised when the word has reached its maximum length."""


class PendingSelectionError(WordBuilderError):
    """Raised for invalid pending-selection actions."""


class EmptyWordError(WordBuilderError):
    """Raised when an empty word is confirmed."""
