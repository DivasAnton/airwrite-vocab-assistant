from enum import StrEnum


class CharacterSource(StrEnum):
    AUTO_ACCEPTED = "AUTO_ACCEPTED"
    USER_SELECTED = "USER_SELECTED"
    MANUAL_INPUT = "MANUAL_INPUT"
