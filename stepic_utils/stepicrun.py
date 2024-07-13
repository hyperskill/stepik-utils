from __future__ import annotations

import argparse
import random
import sys
import unittest
from typing import Final, TYPE_CHECKING

from stepic_utils import utils
from stepic_utils.quiz import BaseQuiz, CodeQuiz, DatasetQuiz, StringQuiz

if TYPE_CHECKING:
    from collections.abc import Sequence

GENERATE_COMMAND: Final = "generate"
SCORE_COMMAND: Final = "score"
SOLVE_COMMAND: Final = "solve"
TEST_COMMAND: Final = "test"
SAMPLE_COMMAND: Final = "sample"
HAS_SOLVE_COMMAND: Final = "has_solve"

DATASET_QUIZ: Final = "dataset"
CODE_QUIZ: Final = "code"
STRING_QUIZ: Final = "string"


class Runner:
    """Runs user code and deals with encodings."""

    def __init__(self, quiz: BaseQuiz, seed: int | None = None) -> None:
        self.seed = seed

        self.encode = utils.encode
        self.decode = utils.decode

        self.quiz = quiz

    def generate(self) -> bytes:
        assert self.seed, "seed should be specified explicitly"
        random.seed(self.seed)
        dataset_clue = self.quiz.generate()
        return self.encode(dataset_clue)

    def score(self, data: bytes) -> bytes:
        reply, clue = self.decode(data)
        result = self.quiz.check(reply, clue)
        return self.encode(result)

    def solve(self, data: bytes) -> bytes:
        dataset = self.decode(data)
        reply = self.quiz.solve(dataset)
        return self.encode(reply)

    def sample(self) -> bytes:
        return self.encode(self.quiz.sample)

    def has_solve(self) -> bytes:
        return self.encode(self.quiz.has_solve)


def read_bin_stdin() -> bytes:
    return sys.stdin.buffer.read()


def write_bin_stdout(data: bytes) -> None:
    sys.stdout.buffer.write(data)


def main(args: Sequence[str] | None = None) -> None:
    parsed_args = parse_arguments(args)

    quiz_class = get_quiz_class(parsed_args)

    quiz = quiz_class.import_quiz(parsed_args.code_path)
    runner = Runner(quiz, seed=parsed_args.seed)

    match parsed_args.command:
        case "generate":
            if not parsed_args.seed:
                print("error: seed should be specified for generate command")  # noqa: T201
                sys.exit()
            # printing binary data in compatible way
            generated = runner.generate()
            sys.stdout.buffer.write(generated)
        case "score":
            binary_input = read_bin_stdin()
            scored = runner.score(binary_input)
            write_bin_stdout(scored)
        case "solve":
            binary_input = read_bin_stdin()
            solved = runner.solve(binary_input)
            write_bin_stdout(solved)
        case "test":
            unittest.main(testLoader=quiz_class.get_test_loader(), module=quiz.module, argv=[sys.argv[0]])
        case "sample":
            sample = runner.sample()
            sys.stdout.buffer.write(sample)
        case "has_solve":
            if quiz_class != CodeQuiz:
                sys.exit(1)
            write_bin_stdout(runner.has_solve())
        case _:
            msg = "unknown command"
            raise AssertionError(msg)


def get_quiz_class(args: argparse.Namespace) -> type[BaseQuiz]:
    match args.type:
        case "dataset":
            quiz_class = DatasetQuiz
        case "code":
            quiz_class = CodeQuiz
        case "string":
            quiz_class = StringQuiz
        case _:
            raise AssertionError
    return quiz_class


def parse_arguments(args: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test or run python exercise")
    parser.add_argument(
        "-c",
        "--command",
        required=True,
        choices=[
            GENERATE_COMMAND,
            SCORE_COMMAND,
            SOLVE_COMMAND,
            TEST_COMMAND,
            SAMPLE_COMMAND,
            HAS_SOLVE_COMMAND,
        ],
    )
    parser.add_argument("-p", "--code-path", default="user_code.py")
    parser.add_argument("-s", "--seed", type=int)
    parser.add_argument("-t", "--type", default=DATASET_QUIZ, choices=[DATASET_QUIZ, CODE_QUIZ, STRING_QUIZ])
    return parser.parse_args(args=args)


if __name__ == "__main__":
    main()
