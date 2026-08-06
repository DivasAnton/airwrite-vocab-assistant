import json
from dataclasses import asdict
from pathlib import Path

from app.utils.config import whole_word_settings


def json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    whole_word_settings.validate()
    print(json.dumps(asdict(whole_word_settings), indent=2, default=json_default))


if __name__ == "__main__":
    main()
