from enum import StrEnum


class Gesture(StrEnum):
    NO_HAND = "NO_HAND"
    UNKNOWN = "UNKNOWN"
    INDEX_ONLY = "INDEX_ONLY"
    TWO_FINGERS = "TWO_FINGERS"
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
