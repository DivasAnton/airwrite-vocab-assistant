CHARACTER_LABELS: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LABEL_TO_INDEX: dict[str, int] = {label: index for index, label in enumerate(CHARACTER_LABELS)}


def normalize_label(label: str) -> str:
    normalized = label.strip().upper()
    if normalized not in LABEL_TO_INDEX:
        raise ValueError(f"Unsupported character label: {label!r}. Expected one of A-Z.")
    return normalized


def label_to_index(label: str) -> int:
    return LABEL_TO_INDEX[normalize_label(label)]


def index_to_label(index: int) -> str:
    if not 0 <= index < len(CHARACTER_LABELS):
        raise ValueError(f"Label index must be between 0 and 25, got {index}")
    return CHARACTER_LABELS[index]


def labels_payload() -> dict[str, object]:
    return {"version": 1, "labels": list(CHARACTER_LABELS)}
