LETTER_IDENTITY_LABELS: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz")


def identity_to_index(identity: str) -> int:
    if identity not in LETTER_IDENTITY_LABELS:
        raise ValueError("identity must be one lowercase letter from a to z")
    return LETTER_IDENTITY_LABELS.index(identity)


def index_to_identity(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("identity index must be an integer from 0 to 25")
    if index < 0 or index >= len(LETTER_IDENTITY_LABELS):
        raise ValueError("identity index must be between 0 and 25")
    return LETTER_IDENTITY_LABELS[index]


def identity_labels_payload() -> dict[str, object]:
    return {"version": 1, "labels": list(LETTER_IDENTITY_LABELS)}


def display_labels_payload(*, uppercase: bool) -> dict[str, object]:
    labels = [label.upper() if uppercase else label for label in LETTER_IDENTITY_LABELS]
    return {"version": 1, "labels": labels}
