from enum import StrEnum


class PredictionStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    UNCERTAIN = "UNCERTAIN"
    SKIPPED_EMPTY = "SKIPPED_EMPTY"
    FAILED = "FAILED"
