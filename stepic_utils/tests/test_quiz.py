from __future__ import annotations

import textwrap
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import pytest

from stepic_utils.quiz import BaseQuiz, check_signatures, CodeQuiz, DatasetQuiz, StringQuiz
from stepic_utils.tests.utils import assert_fail_with_message

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T", bound=BaseQuiz)


def get_quiz(name: str | None = None, code: str | None = None, quiz_cls: type[T] = DatasetQuiz) -> T:
    if name is not None:
        examples_dir = Path(__file__).parent / "examples"
        path = examples_dir / name
        return quiz_cls.import_quiz(path)  # type: ignore[return-value]

    if code is not None:
        quiz_module = module_from_spec(ModuleSpec("quiz", None))
        exec(code, {}, quiz_module.__dict__)  # noqa: S102
        return quiz_cls.import_quiz(quiz_module)  # type: ignore[return-value]

    msg = "Either `name` or `code` should be specified."
    raise ValueError(msg)


def test_ab() -> None:
    quiz: CodeQuiz = get_quiz("ab.py")

    dataset, _clue = quiz.generate()

    assert "file" in dataset


def test_ab_dict() -> None:
    quiz: CodeQuiz = get_quiz("ab_dict.py")

    dataset, _clue = quiz.generate()

    assert "file" not in dataset


def test_hints() -> None:
    quiz: CodeQuiz = get_quiz("hints.py")

    assert "bigger" in quiz.check("32", "")[1]
    assert "smaller" in quiz.check("52", "")[1]
    assert quiz.check("42", "")[0] == 1


@pytest.mark.parametrize(
    "name", ["ab.py", "ab_dict.py", "divisors.py", "hints.py", "even_numbers.py", "fib.py"]
)
def test_all(name: str) -> None:
    quiz: CodeQuiz = get_quiz(name)

    assert quiz.self_check(), f"{name} failed"


def test_float() -> None:
    quiz: CodeQuiz = get_quiz("ab_float.py")

    dataset, _clue = quiz.generate()

    assert "file" in dataset


def test_score_rounding() -> None:
    quiz: CodeQuiz = get_quiz("rounding_errors.py")

    assert quiz.check("1", "1")[0]
    assert not quiz.check("1", "2")[0]


@pytest.mark.parametrize("name", ["string_quiz_with_solve.py", "string_quiz_without_solve.py"])
def test_self_check_success(name: str) -> None:
    quiz = get_quiz(name, quiz_cls=StringQuiz)

    assert quiz.self_check()


def test_self_check_fail() -> None:
    quiz = get_quiz("string_quiz_with_solve_fail.py", quiz_cls=StringQuiz)

    assert not quiz.self_check()


def test_valid() -> None:
    specs = [
        ("f", lambda: None, 0),
        ("f", lambda *args: None, 0),  # noqa: ARG005
        ("f", lambda x: None, 1),  # noqa: ARG005
        ("f", lambda *args: None, 1),  # noqa: ARG005
        ("f", lambda x, *args: None, 1),  # noqa: ARG005
        ("f", lambda x, y: None, 2),  # noqa: ARG005
        ("f", lambda *args: None, 2),  # noqa: ARG005
        ("f", lambda x, *args: None, 2),  # noqa: ARG005
        ("f", lambda x, y, *args: None, 2),  # noqa: ARG005
    ]

    check_signatures(specs)


@pytest.mark.parametrize(
    "f",
    [
        lambda: None,
        lambda x, y: None,  # noqa: ARG005
        lambda x, y, z: None,  # noqa: ARG005
    ],
)
def test_wrong_number_of_args_1_expected(f: Callable[..., object]) -> None:
    specs = [("f", f, 1)]

    with assert_fail_with_message("`f` should accept 1 argument.\n"):
        check_signatures(specs)


@pytest.mark.parametrize(
    "f",
    [
        lambda: None,
        lambda x: None,  # noqa: ARG005
        lambda x, y, z: None,  # noqa: ARG005
    ],
)
def test_wrong_number_of_args_2_expected(f: Callable[..., object]) -> None:
    specs = [("f", f, 2)]

    with assert_fail_with_message("`f` should accept 2 arguments.\n"):
        check_signatures(specs)


def test_generate_required() -> None:
    expected_msg = (
        "Can't import 'generate' from the challenge module.\n"
        "It should export 'generate', 'check' functions.\n"
    )

    with assert_fail_with_message(expected_msg):
        get_quiz(code="", quiz_cls=CodeQuiz)


def test_check_required() -> None:
    code = textwrap.dedent("""
        def generate():
            return []
        """)
    expected_msg = (
        "Can't import 'check' from the challenge module.\n"
        "It should export 'generate', 'check' functions.\n"
    )

    with assert_fail_with_message(expected_msg):
        get_quiz(code=code, quiz_cls=CodeQuiz)


def test_generate_datasets() -> None:
    quiz = get_quiz("code/generate_datasets.py", quiz_cls=CodeQuiz)
    assert quiz

    tests = quiz.generate()

    assert tests == [("2 2\n", "4"), ("5 7\n", "12")]


def test_generate_tuples() -> None:
    quiz = get_quiz("code/generate_tuples.py", quiz_cls=CodeQuiz)

    tests = quiz.generate()

    assert tests == [("2 2\n", "clue:4"), ("5 7\n", "clue:12")]


def test_generate_datasets_without_solve() -> None:
    quiz = get_quiz("code/generate_datasets_without_solve.py", quiz_cls=CodeQuiz)

    tests = quiz.generate()

    assert tests == [("2 2\n", None), ("5 7\n", None)]


def test_generate_tuples_without_solve() -> None:
    quiz = get_quiz("code/generate_tuples_without_solve.py", quiz_cls=CodeQuiz)

    tests = quiz.generate()

    assert tests == [("2 2\n", "clue:4"), ("5 7\n", "clue:12")]


@pytest.mark.parametrize("name", ["code/generate_datasets.py", "code/generate_tuples.py"])
def test_self_check_success_2(name: str) -> None:
    quiz = get_quiz(name, quiz_cls=CodeQuiz)

    assert quiz.self_check()


@pytest.mark.parametrize(
    "name", ["code/generate_datasets_without_solve.py", "code/generate_tuples_without_solve.py"]
)
def test_self_check_without_solve_success(name: str) -> None:
    quiz = get_quiz(name, quiz_cls=CodeQuiz)

    assert quiz.self_check()
