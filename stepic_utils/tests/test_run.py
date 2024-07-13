from __future__ import annotations

from stepic_utils.tests.utils import assert_run_result


def test_has_solve_true():
    args = ("-t", "code", "-p", "code/generate_datasets.py", "-c", "has_solve")

    assert_run_result(args, True)


def test_has_solve_false() -> None:
    args = ("-t", "code", "-p", "code/generate_datasets_without_solve.py", "-c", "has_solve")

    assert_run_result(args, False)
