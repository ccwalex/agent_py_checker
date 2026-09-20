from python_checker.report import (
    Finding,
    Severity,
    Status,
    compute_status,
    exit_code_for_status,
)


def test_compute_status_error_wins():
    findings = [
        Finding("X", Severity.UNCOVERED, "u", "a.py", 1, 1, "r"),
        Finding("Y", Severity.ERROR, "e", "a.py", 2, 1, "r"),
    ]
    assert compute_status(findings) == Status.ERROR


def test_compute_status_uncovered():
    findings = [Finding("X", Severity.UNCOVERED, "u", "a.py", 1, 1, "r")]
    assert compute_status(findings) == Status.UNCOVERED


def test_exit_codes():
    assert exit_code_for_status(Status.OK) == 0
    assert exit_code_for_status(Status.ERROR) == 1
    assert exit_code_for_status(Status.UNCOVERED) == 2
