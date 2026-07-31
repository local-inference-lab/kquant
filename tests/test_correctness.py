import pytest

from kquant.correctness import (
    compare_probe_responses,
    parse_env_assignments,
    probe_payload,
)
from scripts.run_k3_correctness import _python_executable


def _response(
    token_ids: list[int],
    token_logprobs: list[float | None],
) -> dict:
    return {
        "choices": [
            {
                "text": "ignored",
                "token_ids": token_ids,
                "logprobs": {
                    "tokens": [f"t{index}" for index in range(len(token_logprobs))],
                    "token_logprobs": token_logprobs,
                },
            }
        ]
    }


def test_probe_payload_requests_stable_token_identity() -> None:
    payload = probe_payload(model_name="model", max_tokens=7, logprobs=11)

    assert payload["temperature"] == 0
    assert payload["seed"] == 0
    assert payload["return_token_ids"] is True
    assert payload["max_tokens"] == 7
    assert payload["logprobs"] == 11


def test_probe_comparison_requires_ids_and_bounded_logprob_delta() -> None:
    left = _response([4, 5], [None, -1.0, -2.0])
    right = _response([4, 5], [None, -1.02, -1.97])

    report = compare_probe_responses(left, right, logprob_limit=0.04)

    assert report["pass"] is True
    assert report["token_identity"] == "token_ids"
    assert report["same_generated_tokens"] is True
    assert report["max_abs_logprob_delta"] == pytest.approx(0.03)


def test_probe_comparison_rejects_different_generated_ids() -> None:
    left = _response([4, 5], [None, -1.0])
    right = _response([4, 6], [None, -1.0])

    report = compare_probe_responses(left, right)

    assert report["pass"] is False
    assert report["same_generated_tokens"] is False


def test_parse_env_assignments_rejects_shell_syntax() -> None:
    assert parse_env_assignments(["A=1", "B_C=x=y"]) == {
        "A": "1",
        "B_C": "x=y",
    }
    with pytest.raises(ValueError):
        parse_env_assignments(["BAD-NAME=value"])


def test_python_executable_preserves_venv_symlink(tmp_path) -> None:
    target = tmp_path / "base-python"
    target.touch()
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(target)

    assert _python_executable(venv_python) == venv_python
