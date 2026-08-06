from enum import StrEnum


class SegmentationStatus(StrEnum):
    SUCCESS = "SUCCESS"
    EMPTY = "EMPTY"
    AMBIGUOUS = "AMBIGUOUS"
    TOO_FEW_SEGMENTS = "TOO_FEW_SEGMENTS"
    TOO_MANY_SEGMENTS = "TOO_MANY_SEGMENTS"
    FAILED = "FAILED"
