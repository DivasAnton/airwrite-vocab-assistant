from enum import StrEnum


class DrawingState(StrEnum):
    IDLE = "IDLE"
    READY = "READY"
    WRITING = "WRITING"
    PAUSED = "PAUSED"
    DONE = "DONE"
    CLEAR = "CLEAR"
    ERROR = "ERROR"
