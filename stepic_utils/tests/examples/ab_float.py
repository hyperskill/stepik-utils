from __future__ import annotations

import random

from stepic_common import equal_floats, nice

LOW = 1
HIGH = 10


def generate():
    a = random.randrange(LOW, HIGH)
    b = random.randrange(LOW, HIGH)
    return nice(a, b)


def solve(dataset):
    a, b = map(int, dataset.split())
    return str(a + b)


check = equal_floats
