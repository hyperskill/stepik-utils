from __future__ import annotations


def generate():
    return [("2 2\n", "clue:4"), ("5 7\n", "clue:12")]


def check(reply, clue):
    assert clue in {"clue:4", "clue:12"}
    return reply == clue[5:]
