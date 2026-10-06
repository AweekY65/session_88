"""Z-buffer occlusion tests with overlapping triangles."""
import numpy as np

from softrender import Rasterizer


def _quad(z, color):
    positions = [[-1, -1, z], [1, -1, z], [1, 1, z], [-1, 1, z]]
    indices = [[0, 1, 2], [0, 2, 3]]
    colors = [color] * 4
    return positions, indices, colors


def test_nearest_surface_wins_regardless_of_order():
    red = [1.0, 0.0, 0.0]
    blue = [0.0, 0.0, 1.0]
    near_pos, near_idx, near_col = _quad(-0.5, red)   # smaller ndc z = nearer
    far_pos, far_idx, far_col = _quad(0.5, blue)

    for order in ("near_first", "far_first"):
        r = Rasterizer(16, 16)
        draws = {
            "near_first": [(near_pos, near_idx, near_col),
                           (far_pos, far_idx, far_col)],
            "far_first": [(far_pos, far_idx, far_col),
                          (near_pos, near_idx, near_col)],
        }[order]
        for pos, idx, col in draws:
            r.draw(positions=pos, indices=idx, colors=col)
        center = r.framebuffer.color[8, 8]
        np.testing.assert_allclose(center, red, atol=1e-9,
                                   err_msg=f"order={order}")
        np.testing.assert_allclose(r.framebuffer.depth[8, 8], -0.5, atol=1e-9)


def test_partial_overlap():
    # Red far triangle (z=0.5) and green near triangle (z=-0.5) overlap in a
    # diagonal band; each must keep its own non-overlapped region, and the
    # nearer green must win inside the overlap.
    r = Rasterizer(32, 32)
    r.draw(positions=[[-1, -1, 0.5], [0.5, -1, 0.5], [-1, 1, 0.5]],
           indices=[[0, 1, 2]], colors=[[1, 0, 0]] * 3)
    r.draw(positions=[[-0.5, -1, -0.5], [1, -1, -0.5], [1, 1, -0.5]],
           indices=[[0, 1, 2]], colors=[[0, 1, 0]] * 3)
    fb = r.framebuffer
    # (px=4, py=20): ndc (-0.719, -0.281) -> red only.
    np.testing.assert_allclose(fb.color[20, 4], [1, 0, 0], atol=1e-9)
    # (px=30, py=4): ndc (0.906, 0.719) -> green only.
    np.testing.assert_allclose(fb.color[4, 30], [0, 1, 0], atol=1e-9)
    # (px=15, py=23): ndc (-0.031, -0.469) -> overlap, nearer green wins.
    np.testing.assert_allclose(fb.color[23, 15], [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(fb.depth[23, 15], -0.5, atol=1e-9)
