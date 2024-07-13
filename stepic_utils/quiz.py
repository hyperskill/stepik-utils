from __future__ import annotations

import importlib
import itertools
import numbers
import sys
import traceback
import types
import unittest
from functools import wraps
from inspect import signature
from os import PathLike
from pathlib import Path
from typing import NoReturn, TYPE_CHECKING

from stepic_utils import utils

if TYPE_CHECKING:
    from builtins import function
    from collections.abc import Callable


def import_module(path: str | PathLike[str]) -> types.ModuleType:
    path = Path(path).resolve()
    module_dir = str(path.parent)
    module_name = path.stem
    sys.path.insert(0, module_dir)
    return importlib.import_module(module_name)


def check_signatures(
    specs: list[tuple[str, Callable[..., object], int]] | list[tuple[str, function, int]],
) -> None:
    """Check if a function is callable and has a required number of arguments.

    :param specs: [(name, function, expected number of args)]
    """
    for name, f, n_arguments in specs:
        if not callable(f):
            fail_with_message(f"`{name}` is not callable.")
        try:
            signature(f).bind(*[""] * n_arguments)
        except TypeError:
            fail_with_message(
                "`{name}` should accept {n_arguments} argument{s}.".format(
                    name=name, n_arguments=n_arguments, s="" if n_arguments == 1 else "s"
                )
            )


class BaseQuiz:
    export_attrs = ["check", "solve"]

    def __init__(
        self,
        module: types.ModuleType,
        generate_fun: function,
        solve_fun: function,
        check_fun: function,
        tests: list[tuple[str, str, str]],
    ) -> None:
        self.module = module
        self.generate = self.wrap_generate(generate_fun)
        self.solve = self.wrap_solve(solve_fun) if solve_fun is not None else None
        self.check = self.wrap_check(check_fun)
        self.tests = self.clean_tests(tests)
        if self.tests:
            dataset, _clue, reply = self.tests[0]
        else:
            dataset, _clue, reply = "", "", ""
        self.sample = (dataset, reply)

    @classmethod
    def import_quiz(cls, path_or_module: str | Path | types.ModuleType) -> BaseQuiz:
        """Loads quiz from module specified by path.
        Module should export `generate`, `solve` and `check`.
        """
        if isinstance(path_or_module, str | PathLike):
            module = import_module(path_or_module)
        else:
            assert isinstance(path_or_module, types.ModuleType)
            module = path_or_module

        no_function_msg = (
            "Can't import '{}' from the challenge module.\n" "It should export {a}{funcs} function{s}."
        )
        for attr in cls.export_attrs:
            if not hasattr(module, attr):
                fail_with_message(
                    no_function_msg.format(
                        attr,
                        a="a " if len(cls.export_attrs) == 1 else "",
                        funcs=", ".join(f"'{a}'" for a in cls.export_attrs),
                        s="s" if len(cls.export_attrs) > 1 else "",
                    )
                )

        generate = getattr(module, "generate", None)
        solve = getattr(module, "solve", None)
        check = module.check
        tests = getattr(module, "tests", [])
        return cls(module, generate, solve, check, tests)

    @classmethod
    def load_tests(cls, module: types.ModuleType) -> QuizModuleTest:
        return QuizModuleTest(cls, module)

    @classmethod
    def get_test_loader(cls) -> QuizTestLoader:
        return QuizTestLoader(cls)

    def wrap_generate(
        self, generate: Callable[[], tuple[dict[str, object], object]]
    ) -> Callable[[], tuple[dict[str, object], object]]:
        @wraps(generate)
        def f() -> tuple[dict[str, object], object]:
            ret = call_user_code(generate)
            if isinstance(ret, tuple):
                try:
                    dataset, clue = ret
                except ValueError as e:
                    fail_with_message(
                        "generate() returned a tuple but it's length is not 2.\n"
                        "generate should return either a dataset "
                        "or a (dataset, clue) tuple."
                    )
                    raise AssertionError from e
            else:
                dataset = ret
                clue = self.solve(dataset)
            dataset = self.clean_dataset(dataset)
            clue = self.clean_clue(clue)
            return dataset, clue

        return f

    def wrap_solve(self, solve: Callable[[object], str]) -> Callable[[object], str]:
        @wraps(solve)
        def f(dataset: object) -> str:
            if isinstance(dataset, dict) and "file" in dataset and len(dataset) == 1:
                dataset = dataset["file"]
            return self.clean_answer(call_user_code(solve, dataset))

        return f

    def wrap_check(
        self, check: Callable[[str, object], tuple[float, str]]
    ) -> Callable[[str, object], tuple[float, str]]:
        @wraps(check)
        def f(reply: str, clue: object) -> tuple[float, str]:
            ret = call_user_code(check, reply, clue)
            if isinstance(ret, tuple):
                try:
                    score_value, hint = ret
                except ValueError as e:
                    fail_with_message(
                        "check() returned a tuple but it's length is not 2.\n"
                        "check should return either a score or a (score, hint) tuple."
                    )
                    raise AssertionError from e
            else:
                score_value = ret
                hint = ""
            score_value = self.clean_score(score_value)
            hint = self.clean_hint(hint)
            return score_value, hint

        return f

    @staticmethod
    def clean_dataset(dataset: object) -> dict[str, object] | str:
        if not isinstance(dataset, dict | str | bytes):
            msg = "dataset should be one of (dict, str, bytes) instead of {}"
            fail_with_message(msg.format(dataset))

        if isinstance(dataset, str | bytes):
            dataset = {"file": dataset}
        return dataset

    @staticmethod
    def clean_clue(clue: object) -> object:
        try:
            return utils.decode(utils.encode(clue))
        except (TypeError, ValueError):
            msg = "clue is not serializable: {}"
            fail_with_message(msg.format(clue))

    @staticmethod
    def clean_answer(answer: object) -> str:
        if not isinstance(answer, str):
            msg = "answer should be a str instead of {}"
            fail_with_message(msg.format(answer))
        return answer

    @staticmethod
    def clean_score(score: float | numbers.Real) -> float:
        def is_within_delta(x: float, target: float) -> bool:
            return abs(x - target) < delta

        delta = 1e-6
        if not (isinstance(score, numbers.Real) and (0 - delta < score < 1 + delta)):
            fail_with_message("score should be a number in range [0, 1]")

        score = min(1, max(0, score))
        if is_within_delta(score, 0):
            score = 0
        if is_within_delta(score, 1):
            score = 1
        return score

    @staticmethod
    def clean_hint(hint: str) -> str:
        if not isinstance(hint, str):
            fail_with_message("hint should be a str")
        return hint

    @staticmethod
    def clean_tests(tests: list[tuple[str, object, str]]) -> list[tuple[str, object, str]]:
        msg = "`tests` should be a list of triples: [(dataset, clue, reply)]"
        if not isinstance(tests, list):
            fail_with_message(msg)
        for test in tests:
            if not isinstance(test, tuple) or len(test) != 3:
                fail_with_message(msg)
            dataset, clue, reply = test
            if not (isinstance(dataset, str)):
                fail_with_message(
                    f"dataset in `tests` should be a string instead of "
                    f"{type(dataset)}\n{(dataset, clue, reply)}"
                )
            if not (isinstance(reply, str)):
                fail_with_message(
                    f"reply in `tests` should be a string instead of "
                    f"{type(reply)}\n{(dataset, clue, reply)}"
                )
        return tests


class DatasetQuiz(BaseQuiz):
    def __init__(
        self,
        module: types.ModuleType,
        generate_fun: function,
        solve_fun: function,
        check_fun: function,
        tests: list[tuple[str, str, str]],
    ) -> None:
        if generate_fun is None:
            check_signatures([("solve", solve_fun, 0), ("check", check_fun, 1)])

            def generate() -> tuple[dict[str, object], str]:
                return {}, ""

            def solve(dataset: dict[str, object]) -> str:
                return solve_fun()

            def check(reply, clue: str) -> tuple[float, str]:
                return check_fun(reply)
        else:
            check_signatures(
                [("generate", generate_fun, 0), ("solve", solve_fun, 1), ("check", check_fun, 2)]
            )
            generate, solve, check = generate_fun, solve_fun, check_fun
        super().__init__(module, generate, solve, check, tests)

    def self_check(self) -> bool:
        dataset, clue = self.generate()
        answer = self.solve(dataset)
        score, _hint = self.check(answer, clue)
        return score == 1


class CodeQuiz(BaseQuiz):
    export_attrs = ["generate", "check"]

    def __init__(
        self,
        module: types.ModuleType,
        generate_fun: function,
        solve_fun: function,
        check_fun: function,
        tests: list[tuple[str, str, str]],
    ) -> None:
        signature_specs = [("generate", generate_fun, 0), ("check", check_fun, 2)]
        self.has_solve = solve_fun is not None
        if self.has_solve:
            signature_specs.append(("solve", solve_fun, 1))
        check_signatures(signature_specs)
        super().__init__(module, generate_fun, solve_fun, check_fun, tests)

    def wrap_generate(
        self, generate: Callable[[], list[str | tuple[str, str]]]
    ) -> Callable[[], list[tuple[str, str]] | None]:
        @wraps(generate)
        def f() -> list[tuple[str, str]] | None:
            ret = call_user_code(generate)
            if not isinstance(ret, list):
                fail_with_message(f"generate() should return a list instead of {ret}")

            def is_dataset(x):
                return isinstance(x, str)

            def is_dataset_and_clue(x):
                return isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str)

            manual = [(t[0], t[1]) for t in self.tests]
            if all(map(is_dataset, ret)):
                generated_tests = [
                    (
                        self.clean_dataset(dataset),
                        self.clean_clue(self.solve(dataset)) if self.has_solve else None,
                    )
                    for dataset in ret
                ]
                return manual + generated_tests
            if all(map(is_dataset_and_clue, ret)):
                return manual + [
                    (self.clean_dataset(dataset), self.clean_clue(clue)) for dataset, clue in ret
                ]
            fail_with_message(
                "generate() should return list of dataset or list of pairs "
                f"(dataset, clue) instead of {ret}"
            )
            return None

        return f

    @staticmethod
    def clean_dataset(dataset: object) -> str:
        if not isinstance(dataset, str):
            fail_with_message(f"dataset should be a str instead of {dataset}")
        return dataset

    def self_check(self) -> bool:
        def is_correct(dataset: dict[str, object], clue: object) -> bool:
            if not self.has_solve:
                self.check("", clue)
                return True
            answer = self.solve(dataset)
            score, _hint = self.check(answer, clue)
            return score == 1

        test_cases = self.generate()
        return all(itertools.starmap(is_correct, test_cases))


class StringQuiz(BaseQuiz):
    export_attrs = ["check"]

    def __init__(
        self,
        module: types.ModuleType,
        generate_fun: function,
        solve_fun: function,
        check_fun: function,
        tests: list[tuple[str, str, str]],
    ) -> None:
        def generate() -> tuple[dict[str, object], str]:
            return {}, ""

        def check(reply: str, clue: object) -> tuple[float, str]:
            return check_fun(reply)

        self.without_solve = solve_fun is None
        if self.without_solve:
            check_signatures([("check", check_fun, 1)])

            def solve(dataset):
                return ""
        else:
            check_signatures([("solve", solve_fun, 0), ("check", check_fun, 1)])

            def solve(dataset):
                return solve_fun()

        super().__init__(module, generate, solve, check, tests)

    def self_check(self) -> bool:
        answer = self.solve(None)
        score, _hint = self.check(answer, None)
        return self.without_solve or score == 1


class QuizTestLoader(unittest.TestLoader):
    def __init__(self, quiz_cls: type[BaseQuiz]) -> None:
        self.quiz_cls = quiz_cls
        super().__init__()

    def loadTestsFromModule(
        self, module: types.ModuleType, use_load_tests: bool = True
    ) -> unittest.TestSuite:
        suite = super().loadTestsFromModule(module)
        suite.addTest(self.quiz_cls.load_tests(module))
        return suite


class QuizModuleTest(unittest.TestCase):
    def __init__(
        self, quiz_cls: type[BaseQuiz], module: types.ModuleType, method_name: str = "runTest"
    ) -> None:
        super().__init__(method_name)
        self.quiz = quiz_cls.import_quiz(module)

    def runTest(self) -> None:
        self.testSamples()
        self.testSolve()

    def testSamples(self) -> None:
        for _dataset, clue, reply in self.quiz.tests:
            msg = "\nscore(reply, clue) != 1!\nscore({}, {}) == {}"
            score, _ = self.quiz.check(reply, clue)
            assert score == 1, msg.format(reply, clue, score)

    def testSolve(self) -> None:
        for dataset, clue, _reply in self.quiz.tests:
            computed_reply = self.quiz.solve(dataset)
            msg = "\nscore(solve(dataset), clue) != 1!\nscore({}, {}) == {}"
            score, _ = self.quiz.check(computed_reply, clue)
            assert score == 1, msg.format(computed_reply, clue, score)


def fail_with_message(message: str) -> NoReturn:
    print(message, file=sys.stderr)  # noqa: T201
    sys.exit(-1)


def call_user_code(function: Callable[..., object], *args, **kwargs) -> object:
    try:
        return function(*args, **kwargs)
    except Exception:
        traceback.print_exc()
        fail_with_message("Quiz failed with exception!")
