"""Image output: binary PPM (P6) and PNG, both pure stdlib writers."""

import struct
import zlib

import numpy as np


def _as_uint8(rgb):
    a = np.asarray(rgb)
    if a.dtype != np.uint8:
        a = np.clip(a, 0.0, 1.0)
        a = (a * 255.0).round().astype(np.uint8)
    return a


def write_ppm(path, rgb):
    """Write an (H, W, 3) image as a binary PPM (P6) file."""
    a = _as_uint8(rgb)
    h, w = a.shape[0], a.shape[1]
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h))
        f.write(a.tobytes())


def write_png(path, rgb):
    """Write an (H, W, 3) image as an 8-bit RGB PNG (no external deps)."""
    a = _as_uint8(rgb)
    h, w = a.shape[0], a.shape[1]

    def chunk(tag, payload):
        out = struct.pack(">I", len(payload)) + tag + payload
        out += struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        return out

    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(raw)))
        f.write(chunk(b"IEND", b""))
