"""End-to-end render + image file output tests (PPM / PNG).

Renders the shared deterministic scene, writes PPM and PNG files, and
verifies fixed pixel values plus SHA-256 hashes of the encoded bytes so any
regression in the pipeline is caught. Runs entirely in the terminal.
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from scene import render_scene  # noqa: E402

from softrender import encode_png, save_png, save_ppm, to_uint8  # noqa: E402

# Golden hashes of the deterministic 64x64 test scene.
PPM_SHA256 = "ff7a6b1dd4916b45feae86d19a7f342194969bb068ff2d81f8a6039c756eb4f9"
PNG_SHA256 = "24a7748c70ce7d2217a6065f2621c46738befde21f3c436893ec09ace0e840a0"

# Fixed pixels of the scene (row, col) -> expected RGB uint8.
EXPECTED_PIXELS = {
    (16, 32): (0, 0, 0),          # background above the horizon
    (24, 28): (230, 26, 26),      # red triangle
    (40, 32): (169, 169, 169),    # bilinear-filtered checker cell
    (56, 16): (136, 136, 136),    # bilinear-filtered checker cell
    (48, 48): (205, 205, 205),    # bilinear-filtered checker cell
}


def test_fixed_pixels():
    r = render_scene()
    img = to_uint8(r.framebuffer.color)
    for (py, px), expected in EXPECTED_PIXELS.items():
        got = tuple(int(c) for c in img[py, px])
        assert got == expected, f"pixel ({py},{px}): got {got}, want {expected}"


def test_green_triangle_occludes_red():
    r = render_scene()
    img = to_uint8(r.framebuffer.color)
    # The near green triangle overlaps the far red one around row 22-26.
    green_px = img[24, 34]
    assert green_px[1] > 200 and green_px[0] < 80


def test_ppm_output(tmp_path):
    r = render_scene()
    path = tmp_path / "scene.ppm"
    save_ppm(str(path), r.framebuffer.color)
    data = path.read_bytes()
    assert data.startswith(b"P6\n64 64\n255\n")
    assert len(data) == len(b"P6\n64 64\n255\n") + 64 * 64 * 3
    assert hashlib.sha256(data).hexdigest() == PPM_SHA256


def test_png_output(tmp_path):
    r = render_scene()
    png = encode_png(r.framebuffer.color)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert hashlib.sha256(png).hexdigest() == PNG_SHA256
    path = tmp_path / "scene.png"
    save_png(str(path), r.framebuffer.color)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PNG_SHA256


def test_depth_buffer_values_sane():
    r = render_scene()
    depth = r.framebuffer.depth
    covered = np.isfinite(depth)
    assert covered.sum() > 2000
    assert (depth[covered] >= -1.0).all()
    assert (depth[covered] <= 1.0).all()
    assert np.isfinite(r.framebuffer.color).all()
