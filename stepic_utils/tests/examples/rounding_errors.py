from __future__ import annotations

import random

from stepic_common import nice

LOW = 1
HIGH = 10


def generate():
    a = random.randrange(LOW, HIGH)
    b = random.randrange(LOW, HIGH)
    return nice(a, b)


def solve(dataset):
    a, b = map(int, dataset.split())
    return str(a + b)


def check(reply, hint):
    d = 1e-8
    return 1 + d if int(reply) == int(hint) else d
