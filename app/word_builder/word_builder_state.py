from enum import StrEnum


class WordBuilderState(StrEnum):
    EMPTY = "EMPTY"
    BUILDING = "BUILDING"
    AWAITING_SELECTION = "AWAITING_SELECTION"
    CONFIRMED = "CONFIRMED"
