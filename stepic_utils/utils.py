from __future__ import annotations

import base64
import datetime
import decimal
import json
import pickle  # noqa: S403
import time
from json.decoder import WHITESPACE
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def _encode_pickle(data):
    """:returns: bytes -- encoded data"""
    return base64.b64encode(
        # to be compatible with Python2
        pickle.dumps(data, protocol=2)
    )


def _decode_pickle(data):
    """:type data: bytes."""
    return pickle.loads(base64.b64decode(data))  # noqa: S301


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return {"_serialized.decimal": str(o)}
        if isinstance(o, datetime.datetime):
            # doesn't preserve timezone
            return {"_serialized.datetime": o.timestamp()}
        if isinstance(o, datetime.date):
            return {"_serialized.date": int(time.mktime(o.timetuple()))}
        if isinstance(o, datetime.timedelta):
            return {"_serialized.timedelta": o.total_seconds()}
        return super().default(o)


class JSONDecoder(json.JSONDecoder):
    def decode(self, s: str, _w: Callable[..., object] = WHITESPACE.match) -> object:
        obj = super().decode(s, _w)
        if not isinstance(obj, dict):
            return obj
        if "_serialized.decimal" in obj:
            return decimal.Decimal(obj["_serialized.decimal"])
        if "_serialized.datetime" in obj:
            return datetime.datetime.fromtimestamp(obj["_serialized.datetime"])
        if "_serialized.date" in obj:
            return datetime.date.fromtimestamp(obj["_serialized.date"])
        if "_serialized.timedelta" in obj:
            return datetime.timedelta(seconds=obj["_serialized.timedelta"])
        return obj


json_encoder = JSONEncoder()
json_decoder = JSONDecoder()


def _encode_json(data: object) -> bytes:
    return json_encoder.encode(data).encode()


def _decode_json(data: bytes) -> object:
    return json_decoder.decode(data.decode())


encode = _encode_json
decode = _decode_json
