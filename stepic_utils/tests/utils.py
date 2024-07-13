from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO, StringIO
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from stepic_utils.stepicrun import main
from stepic_utils.utils import decode

if TYPE_CHECKING:
    from collections.abc import Sequence


@contextmanager
def assert_fail_with_message(expected_message: str) -> None:
    fake_stderr = StringIO()
    with patch("sys.stderr", fake_stderr), pytest.raises(SystemExit):
        yield
    stderr = fake_stderr.getvalue()
    assert stderr == expected_message


def assert_run_result(args: Sequence[str], expected_result: object) -> None:
    fake_buffer = BytesIO()
    with patch("sys.stdout") as mock_stdout:
        mock_stdout.buffer = fake_buffer
        main(args=args)
    result = decode(fake_buffer.getvalue())
    assert result == expected_result
