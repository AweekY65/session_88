"""Integration test: render a small scene, write PPM + PNG to disk,
verify fixed pixels and a golden hash of the frame. All checks run in
the terminal; no window is ever opened."""

import hashlib
import zlib

import numpy as np

from softrenderer import Renderer, Texture, look_at, write_png, write_ppm

# Golden hash of the quantized RGB frame (see test_scene_golden_hash).
GOLDEN_SHA256 = "e353dbbab39b7163805fbf01cbcf49a3549386053d398bce0390635478478cd3"

RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)


def render_scene():
    """Textured floor + two overlapping triangles (depth test demo)."""
    r = Renderer(96, 96, fovy=np.radians(60.0), near=0.1, far=100.0,
                 clear_color=(0.1, 0.1, 0.15))
    view = look_at(eye=(0, 1.0, 3.0), center=(0, 0, 0), up=(0, 1, 0))

    # checkerboard floor
    floor_pos = np.array([
        [-4, -1, 2], [4, -1, 2], [4, -1, -6], [-4, -1, -6],
    ], dtype=float)
    floor_uv = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
    floor_col = np.ones((4, 3))
    floor_idx = np.array([[0, 1, 2], [0, 2, 3]])
    checker = Texture.checker(64, 64, 8, 8, (0.9, 0.9, 0.9), (0.2, 0.2, 0.2))
    r.render(floor_pos, floor_uv, floor_col, floor_idx, view=view,
             texture=checker, sample_mode="bilinear", wrap="repeat")

    # two overlapping triangles: red is nearer and must win
    tri_pos = np.array([
        [-1.2, -0.8, 0.5], [0.6, -0.8, 0.5], [-0.3, 1.0, 0.5],   # red, near
        [-0.6, -0.6, -0.5], [1.2, -0.6, -0.5], [0.3, 1.2, -0.5],  # green, far
    ], dtype=float)
    tri_uv = np.zeros((6, 2))
    tri_col = np.array([RED, RED, RED, GREEN, GREEN, GREEN], dtype=float)
    tri_idx = np.array([[0, 1, 2], [3, 4, 5]])
    r.render(tri_pos, tri_uv, tri_col, tri_idx, view=view)
    return r


def test_scene_fixed_pixels():
    r = render_scene()
    img = r.fb.color
    # background (top-left corner, above the horizon)
    assert np.allclose(img[5, 5], (0.1, 0.1, 0.15))
    # overlap region: red triangle is nearer -> red wins
    assert np.allclose(img[44, 44], RED)
    # green-only region
    assert np.allclose(img[30, 60], GREEN)
    # floor region: must be one of the two checker colors
    floor_px = img[88, 48]
    assert (np.allclose(floor_px, (0.9, 0.9, 0.9), atol=0.05)
            or np.allclose(floor_px, (0.2, 0.2, 0.2), atol=0.05))


def test_scene_golden_hash():
    r = render_scene()
    digest = hashlib.sha256(r.fb.color_uint8().tobytes()).hexdigest()
    assert digest == GOLDEN_SHA256


def test_ppm_output(tmp_path):
    r = render_scene()
    path = tmp_path / "scene.ppm"
    r.fb.save_ppm(str(path))
    data = path.read_bytes()
    assert data.startswith(b"P6\n96 96\n255\n")
    payload = data[len(b"P6\n96 96\n255\n"):]
    assert len(payload) == 96 * 96 * 3
    # pixel (0, 0) is the clear color
    assert tuple(payload[0:3]) == (26, 26, 38)  # 0.1*255=26, 0.15*255=38
    # hash of the file payload matches the in-memory frame
    assert hashlib.sha256(payload).hexdigest() == \
        hashlib.sha256(r.fb.color_uint8().tobytes()).hexdigest()


def test_png_output(tmp_path):
    r = render_scene()
    path = tmp_path / "scene.png"
    r.fb.save_png(str(path))
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    # walk the chunks, decompress IDAT and compare against the frame
    pos = 8
    idat = b""
    saw_ihdr = saw_iend = False
    while pos < len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        crc = int.from_bytes(data[pos + 8 + length:pos + 12 + length], "big")
        assert crc == zlib.crc32(tag + payload) & 0xFFFFFFFF
        if tag == b"IHDR":
            saw_ihdr = True
            w, h = int.from_bytes(payload[0:4], "big"), \
                int.from_bytes(payload[4:8], "big")
            assert (w, h) == (96, 96)
        elif tag == b"IDAT":
            idat += payload
        elif tag == b"IEND":
            saw_iend = True
        pos += 12 + length
    assert saw_ihdr and saw_iend
    raw = zlib.decompress(idat)
    stride = 96 * 3 + 1
    assert len(raw) == 96 * stride
    frame = r.fb.color_uint8()
    for y in range(96):
        assert raw[y * stride] == 0  # filter type 0
        assert raw[y * stride + 1:(y + 1) * stride] == frame[y].tobytes()


def test_write_helpers_accept_float_images(tmp_path):
    img = np.zeros((4, 4, 3))
    img[0, 0] = (1.0, 0.0, 0.0)
    p1 = tmp_path / "a.ppm"
    p2 = tmp_path / "a.png"
    write_ppm(str(p1), img)
    write_png(str(p2), img)
    assert p1.read_bytes()[len(b"P6\n4 4\n255\n"):len(b"P6\n4 4\n255\n") + 3] \
        == bytes((255, 0, 0))
