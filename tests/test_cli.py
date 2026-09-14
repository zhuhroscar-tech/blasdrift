import json
import subprocess
import sys

from blasdrift.cli import main


def _run_cli(args, capsys):
    exit_code = main(args)
    captured = capsys.readouterr()
    return exit_code, captured.out


def test_version_flag(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "blasdrift" in out


def test_list_fixtures(capsys):
    exit_code, out = _run_cli(["--list-fixtures"], capsys)
    assert exit_code == 0
    assert "qmcpack_near_zero_dot" in out
    assert "large_magnitude_disparity_sum" in out


def test_json_output_all_fixtures(capsys):
    exit_code, out = _run_cli(["--json", "--no-color"], capsys)
    assert exit_code == 0
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) == 4
    for item in data:
        assert "fixture" in item
        assert "verdict" in item


def test_json_single_fixture(capsys):
    exit_code, out = _run_cli(["--json", "--fixture", "qmcpack_near_zero_dot"], capsys)
    assert exit_code == 0
    data = json.loads(out)
    assert len(data) == 1
    assert data[0]["fixture"] == "qmcpack_near_zero_dot"


def test_text_output_no_color_has_no_ansi_codes(capsys):
    exit_code, out = _run_cli(["--no-color"], capsys)
    assert exit_code == 0
    assert "\033[" not in out


def test_check_drift_exit_code_is_zero_or_one(capsys):
    exit_code, out = _run_cli(["--check-drift", "--no-color"], capsys)
    assert exit_code in (0, 1)


def test_invalid_fixture_name_rejected_by_argparse(capsys):
    import pytest

    with pytest.raises(SystemExit):
        main(["--fixture", "not-a-real-fixture"])


def test_thread_sensitivity_json(capsys):
    exit_code, out = _run_cli(["--thread-sensitivity", "qmcpack_near_zero_dot", "--json"], capsys)
    assert exit_code == 0
    data = json.loads(out)
    assert data["fixture"] == "qmcpack_near_zero_dot"
    assert data["status"] in ("stable", "thread_sensitive", "inconclusive")


def test_console_script_entrypoint_runs():
    # Exercises the packaged entry point exactly as CI/users invoke it,
    # matching the convention used by the other owned CLIs in this fleet.
    proc = subprocess.run(
        [sys.executable, "-m", "blasdrift.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "blasdrift" in proc.stdout
