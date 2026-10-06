"""Triangle coverage and barycentric interpolation tests."""
import numpy as np

from softrender import Rasterizer, barycentric


def test_barycentric_math():
    a = (0.0, 0.0)
    b = (4.0, 0.0)
    c = (0.0, 4.0)
    w = barycentric(1.0, 1.0, a, b, c)
    np.testing.assert_allclose(w, [0.5, 0.25, 0.25], atol=1e-12)
    assert sum(w) == 1.0
    # Degenerate triangle -> None
    assert barycentric(0.5, 0.5, (0, 0), (1, 1), (2, 2)) is None


def test_triangle_coverage():
    # NDC triangle (-1,-1), (-1,1), (1,1) covers the upper-left half of a
    # 4x4 framebuffer (identity transform, w = 1).
    r = Rasterizer(4, 4)
    r.draw(
        positions=[[-1, -1, 0], [-1, 1, 0], [1, 1, 0]],
        indices=[[0, 1, 2]],
        colors=[[1, 0, 0]] * 3,
    )
    fb = r.framebuffer
    red = np.array([1.0, 0.0, 0.0])
    black = np.array([0.0, 0.0, 0.0])
    # Screen-space triangle: (0,4), (0,0), (4,0); inside iff x + y <= 4.
    np.testing.assert_allclose(fb.color[0, 0], red)   # (0.5, 0.5) inside
    np.testing.assert_allclose(fb.color[0, 3], red)   # (3.5, 0.5) on edge
    np.testing.assert_allclose(fb.color[3, 0], red)   # (0.5, 3.5) on edge
    np.testing.assert_allclose(fb.color[3, 3], black)  # (3.5, 3.5) outside
    np.testing.assert_allclose(fb.color[2, 2], black)  # (2.5, 2.5) outside
    assert np.isinf(fb.depth[3, 3])
    assert fb.depth[0, 0] == 0.0


def test_vertex_color_interpolation():
    # Identity transform keeps w = 1, so perspective-correct == linear.
    w = h = 31
    r = Rasterizer(w, h)
    r.draw(
        positions=[[-1, -1, 0], [1, -1, 0], [0, 1, 0]],
        indices=[[0, 1, 2]],
        colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    px, py = 15, 20  # pixel center (15.5, 20.5)
    # Expected color from barycentric weights at the pixel center.
    sx = [(-1 + 1) * w / 2, (1 + 1) * w / 2, (0 + 1) * w / 2]
    sy = [(1 - (-1)) * h / 2, (1 - (-1)) * h / 2, (1 - 1) * h / 2]
    w0, w1, w2 = barycentric(px + 0.5, py + 0.5,
                             (sx[0], sy[0]), (sx[1], sy[1]), (sx[2], sy[2]))
    expected = np.array([w0, w1, w2])
    np.testing.assert_allclose(r.framebuffer.color[py, px], expected, atol=1e-9)
