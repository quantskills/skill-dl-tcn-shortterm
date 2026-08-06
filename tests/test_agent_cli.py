from __future__ import annotations

import json
from pathlib import Path

import pytest

import skill_dl_tcn_shortterm.agent_cli as agent_cli
from skill_dl_tcn_shortterm.agent_cli import main, run_agent_request


def _last_json(captured: str) -> dict[str, object]:
    lines = [line for line in captured.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert isinstance(payload, dict)
    return payload


def test_demo_exposes_one_agent_neutral_json_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["demo"]) == 0
    captured = capsys.readouterr()
    payload = _last_json(captured.out)

    assert payload["status"] == "success"
    assert payload["action"] == "demo"
    assert payload["engine"] == {
        "name": "skill-dl-tcn-shortterm",
        "version": "0.1.0",
    }
    assert payload["errors"] == []
    assert len(str(payload["request_digest"])) == 64


def test_example_then_run_uses_paths_relative_to_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    example_root = tmp_path / "portable-example"
    assert main(["example", "--output-dir", str(example_root)]) == 0
    captured = capsys.readouterr()
    example_payload = _last_json(captured.out)
    assert example_payload["engine"] == {
        "name": "skill-dl-tcn-shortterm",
        "version": "0.1.0",
    }
    assert example_payload["files"] == [
        "samples.parquet",
        "manifest.json",
        "config.json",
        "request.json",
    ]

    result = run_agent_request(example_root / "request.json")

    assert result["status"] == "success"
    assert result["action"] == "run"
    assert result["authoritative_run_manifest"]["model"] == "constant-zero"
    assert (example_root / "runs" / str(result["run_id"]) / "run.json").is_file()


def test_unknown_request_fields_fail_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "action": "run",
                "config_path": "config.json",
                "manifest_path": "manifest.json",
                "output_root": "runs",
                "agent_specific_mode": "codex",
            }
        ),
        encoding="utf-8",
    )

    assert main(["run", "--request", str(request)]) == 2
    captured = capsys.readouterr()
    payload = _last_json(captured.out)
    assert payload["status"] == "failed"
    assert "unknown fields" in str(payload["errors"])


def test_unexpected_engine_errors_still_return_one_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")

    def fail(_: Path) -> dict[str, object]:
        raise RuntimeError("engine canary failure")

    monkeypatch.setattr(agent_cli, "run_agent_request", fail)

    assert main(["run", "--request", str(request)]) == 2
    captured = capsys.readouterr()
    payload = _last_json(captured.out)
    assert payload["status"] == "failed"
    assert payload["errors"] == ["engine canary failure"]


def test_request_and_result_schemas_are_available(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for kind, title in (
        ("request", "TCN Short-Term Agent Request v1"),
        ("result", "TCN Short-Term Agent Result v1"),
    ):
        assert main(["schema", "--kind", kind]) == 0
        captured = capsys.readouterr()
        payload = _last_json(captured.out)
        assert payload["title"] == title


def test_pyproject_registers_flat_console_interface() -> None:
    pyproject = (
        Path(__file__).resolve().parents[1] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert (
        'tcn-shortterm-skill = "skill_dl_tcn_shortterm.agent_cli:main"'
        in pyproject
    )
    assert 'readme = "README.md"' in pyproject
