"""Minimal TensorBoard event-file reader (stdlib only).

Reads TFRecord-framed `Event` protobufs and yields (step, tag, value) for scalar summaries,
handling both `simple_value` and single-float `tensor` encodings.
"""
import struct
import sys
from collections import defaultdict
from pathlib import Path


def _varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf):
    """Yield (field_number, wire_type, value) for one protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _varint(buf, i)
        elif wire == 1:
            val, i = buf[i:i + 8], i + 8
        elif wire == 2:
            ln, i = _varint(buf, i)
            val, i = buf[i:i + ln], i + ln
        elif wire == 5:
            val, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, wire, val


def _tensor_float(buf):
    for f, w, v in _fields(buf):
        if f == 5 and w == 2 and len(v) >= 4:  # packed float_val
            return struct.unpack("<f", v[:4])[0]
        if f == 5 and w == 5:
            return struct.unpack("<f", v)[0]
        if f == 4 and w == 2 and len(v) == 4:  # tensor_content
            return struct.unpack("<f", v)[0]
    return None


def scalars(path):
    data = Path(path).read_bytes()
    i = 0
    while i + 12 <= len(data):
        (length,) = struct.unpack("<Q", data[i:i + 8])
        rec = data[i + 12:i + 12 + length]
        i += 12 + length + 4
        step = 0
        summary = None
        for f, w, v in _fields(rec):
            if f == 2 and w == 0:
                step = v
            elif f == 5 and w == 2:
                summary = v
        if summary is None:
            continue
        for f, w, v in _fields(summary):
            if f != 1 or w != 2:
                continue
            tag, value = None, None
            for vf, vw, vv in _fields(v):
                if vf == 1 and vw == 2:
                    tag = vv.decode("utf-8", "replace")
                elif vf == 2 and vw == 5:
                    value = struct.unpack("<f", vv)[0]
                elif vf == 8 and vw == 2 and value is None:
                    value = _tensor_float(vv)
            if tag is not None and value is not None:
                yield step, tag, value


def run_series(run_dir):
    """Merge all event files in a run directory into {tag: sorted [(step, value)]}."""
    series = defaultdict(dict)
    for f in sorted(Path(run_dir).glob("events.out.tfevents.*")):
        for step, tag, value in scalars(f):
            series[tag][step] = value
    return {t: sorted(d.items()) for t, d in series.items()}


if __name__ == "__main__":
    s = run_series(sys.argv[1])
    for tag, pts in sorted(s.items()):
        print(f"{len(pts):6d}  {tag}  last={pts[-1]}")
