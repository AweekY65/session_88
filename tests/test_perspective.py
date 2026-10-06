"""Perspective-correct interpolation tests.

Ground truth is computed independently by ray/plane intersection in view
space, so the rasterizer's 1/w-weighted interpolation is verified against
real projective geometry rather than its own formula.
"""
import numpy as np

from softrender import Rasterizer, math3d

W = H = 64
# View-space triangle with strongly varying depth.
A = np.array([-1.6, -1.2, -2.0])
B = np.array([1.6, -1.0, -3.0])
C = np.array([0.0, 1.6, -4.0])
UVS = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]


def _render():
    r = Rasterizer(W, H)
    proj = math3d.perspective(np.pi / 2, 1.0, 0.1, 100.0)
    r.draw(
        positions=[A, B, C],
        indices=[[0, 1, 2]],
        uvs=UVS,
        colors=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],  # color.rg == uv
        projection=proj,
    )
    return r


def _ground_truth_uv(px, py):
    """Intersect the view ray through pixel (px, py) with triangle ABC."""
    ndc_x = (px + 0.5) / W * 2.0 - 1.0
    ndc_y = 1.0 - (py + 0.5) / H * 2.0
    d = np.array([ndc_x, ndc_y, -1.0])  # f = 1 for fovy = 90 deg
    n = np.cross(B - A, C - A)
    t = np.dot(n, A) / np.dot(n, d)
    p = d * t
    # Barycentric coordinates of p in 3D.
    v0, v1, v2 = B - A, C - A, p - A
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    l1 = (d11 * d20 - d01 * d21) / denom
    l2 = (d00 * d21 - d01 * d20) / denom
    l0 = 1.0 - l1 - l2
    if min(l0, l1, l2) < -1e-9:
        return None
    return l0 * UVS[0] + l1 * UVS[1] + l2 * UVS[2]


def _naive_screen_linear_uv(px, py):
    """What plain (incorrect) screen-space linear interpolation gives."""
    proj = math3d.perspective(np.pi / 2, 1.0, 0.1, 100.0)
    vp = math3d.viewport(0, 0, W, H)
    pts = []
    for v in (A, B, C):
        clip = proj @ np.append(v, 1.0)
        ndc = clip[:3] / clip[3]
        pts.append((vp @ np.append(ndc, 1.0))[:2])
    (x0, y0), (x1, y1), (x2, y2) = pts
    d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    w0 = ((y1 - y2) * (px + 0.5 - x2) + (x2 - x1) * (py + 0.5 - y2)) / d
    w1 = ((y2 - y0) * (px + 0.5 - x2) + (x0 - x2) * (py + 0.5 - y2)) / d
    w2 = 1.0 - w0 - w1
    return w0 * UVS[0] + w1 * UVS[1] + w2 * UVS[2]


def test_perspective_correct_interpolation():
    r = _render()
    checked = 0
    max_naive_error = 0.0
    for py in range(H):
        for px in range(W):
            if np.isinf(r.framebuffer.depth[py, px]):
                continue
            truth = _ground_truth_uv(px, py)
            assert truth is not None
            got = r.framebuffer.color[py, px][:2]
            np.testing.assert_allclose(got, truth, atol=1e-6)
            naive = _naive_screen_linear_uv(px, py)
            max_naive_error = max(max_naive_error,
                                  float(np.max(np.abs(naive - truth))))
            checked += 1
    assert checked > 500  # triangle covers a good part of the screen
    # The test must actually discriminate: naive linear interpolation would
    # be visibly wrong on this triangle.
    assert max_naive_error > 0.05
