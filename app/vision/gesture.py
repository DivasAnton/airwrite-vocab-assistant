from enum import StrEnum


class Gesture(StrEnum):
    NO_HAND = "NO_HAND"
    UNKNOWN = "UNKNOWN"
    INDEX_ONLY = "INDEX_ONLY"
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
