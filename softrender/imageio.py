"""Image output: binary PPM (P6) and PNG (pure stdlib via zlib), no dependencies."""
import struct
import zlib

import numpy as np


def to_uint8(color_buffer):
    """Float [0,1] HxWx3 buffer -> uint8."""
    return (np.clip(color_buffer, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_ppm(path, color_buffer):
    """Write a binary PPM (P6) image."""
    img = to_uint8(color_buffer)
    h, w = img.shape[:2]
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h))
        f.write(img.tobytes())


def _png_chunk(tag, payload):
    chunk = tag + payload
    return struct.pack(">I", len(payload)) + chunk + struct.pack(
        ">I", zlib.crc32(chunk) & 0xFFFFFFFF
    )


def encode_png(color_buffer):
    """Encode an HxWx3 float buffer as deterministic PNG bytes (RGB, 8-bit)."""
    img = to_uint8(color_buffer)
    h, w = img.shape[:2]
    raw = bytearray()
    stride = w * 3
    flat = img.reshape(h, stride)
    for row in range(h):
        raw.append(0)  # filter type 0 (None) for determinism
        raw.extend(flat[row].tobytes())
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _png_chunk(b"IEND", b"")
    )


def save_png(path, color_buffer):
    with open(path, "wb") as f:
        f.write(encode_png(color_buffer))
