from argparse import Namespace

import pytest

from scripts import collect_custom_data
from scripts.collect_custom_data import normalize_label


@pytest.mark.parametrize(
    ("value", "style", "expected"),
    [("r", "lowercase", "r"), ("R", "lowercase", "r"), ("z", "uppercase", "Z")],
)
def test_normalize_label_matches_requested_writing_style(
    value: str, style: str, expected: str
) -> None:
    assert normalize_label(value, style) == expected


@pytest.mark.parametrize("value", ["", "rr", "1", "é"])
def test_normalize_label_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="one ASCII letter"):
        normalize_label(value, "lowercase")


def test_main_configures_an_exact_uppercase_capture_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    arguments = Namespace(
        label="Z",
        limit=63,
        style=None,
        output_dir=tmp_path,
        save_key="v",
    )
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd, env, check):
        captured.update(command=command, cwd=cwd, env=env, check=check)
        return Namespace(returncode=0)

    monkeypatch.setattr(collect_custom_data, "parse_args", lambda: arguments)
    monkeypatch.setattr(collect_custom_data.subprocess, "run", fake_run)

    assert collect_custom_data.main() == 0
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["DATASET_CAPTURE_INITIAL_LABEL"] == "Z"
    assert environment["DATASET_CAPTURE_WRITING_STYLE"] == "uppercase"
    assert environment["DATASET_CAPTURE_LIMIT"] == "63"
    assert environment["DATASET_CAPTURE_CLEAR_AFTER_SAVE"] == "true"
