from python_checker.ruff_adapter import RUFF_SELECT


def test_ruff_select_pinned():
    assert "F" in RUFF_SELECT
    assert "E9" in RUFF_SELECT
