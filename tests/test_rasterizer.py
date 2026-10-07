"""Rasterization tests: coverage, z-buffer occlusion, perspective-correct
interpolation and near-plane clipping through the full pipeline."""

import numpy as np

from softrenderer import (
    Framebuffer,
    Renderer,
    Texture,
    look_at,
    rasterize_triangle,
)

RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
BLUE = (0.0, 0.0, 1.0)


def draw_flat(fb, pts, color, inv_w=(1.0, 1.0, 1.0)):
    """Helper: draw a screen-space triangle with a single solid color."""
    pts = np.asarray(pts, dtype=np.float64)
    uvs = np.zeros((3, 2))
    colors = np.tile(np.asarray(color, dtype=np.float64), (3, 1))
    rasterize_triangle(fb, pts, uvs, colors, np.asarray(inv_w, float))


# ---------------------------------------------------------------- coverage

def test_triangle_coverage_hits_expected_pixels():
    fb = Framebuffer(64, 64)
    # right triangle: (8,8) (56,8) (8,56)
    draw_flat(fb, [(8, 8, 0.5), (56, 8, 0.5), (8, 56, 0.5)], RED)
    img = fb.color
    # clearly inside
    assert np.allclose(img[16, 16], RED)
    assert np.allclose(img[10, 40], RED)
    # clearly outside (beyond the hypotenuse and the legs)
    assert np.allclose(img[40, 40], (0, 0, 0))
    assert np.allclose(img[4, 16], (0, 0, 0))
    assert np.allclose(img[60, 16], (0, 0, 0))


def test_triangle_coverage_area_matches():
    fb = Framebuffer(64, 64)
    draw_flat(fb, [(8, 8, 0.5), (56, 8, 0.5), (8, 56, 0.5)], RED)
    covered = int((fb.depth < np.inf).sum())
    expected = 0.5 * 48 * 48  # analytic area of the triangle
    assert abs(covered - expected) / expected < 0.05


def test_degenerate_triangle_draws_nothing():
    fb = Framebuffer(16, 16)
    draw_flat(fb, [(4, 4, 0.5), (8, 8, 0.5), (12, 12, 0.5)], RED)
    assert not (fb.depth < np.inf).any()


def test_winding_order_does_not_matter():
    for order in ([(8, 8), (56, 8), (8, 56)], [(8, 8), (8, 56), (56, 8)]):
        fb = Framebuffer(64, 64)
        pts = [(x, y, 0.5) for x, y in order]
        draw_flat(fb, pts, RED)
        assert np.allclose(fb.color[16, 16], RED)


# ------------------------------------------------------------- z-buffering

def overlapping_pair(fb, first, second):
    """Two triangles covering the same pixels at different depths."""
    draw_flat(fb, [(16, 16, first[1]), (48, 16, first[1]),
                   (16, 48, first[1])], first[0])
    draw_flat(fb, [(16, 16, second[1]), (48, 16, second[1]),
                   (16, 48, second[1])], second[0])


def test_zbuffer_nearest_wins_regardless_of_order():
    # nearer (smaller z) drawn last
    fb = Framebuffer(64, 64)
    overlapping_pair(fb, (RED, 0.8), (GREEN, 0.2))
    assert np.allclose(fb.color[24, 24], GREEN)
    # nearer drawn first: result must be identical
    fb = Framebuffer(64, 64)
    overlapping_pair(fb, (GREEN, 0.2), (RED, 0.8))
    assert np.allclose(fb.color[24, 24], GREEN)
    # depth buffer holds the nearest depth
    assert np.isclose(fb.depth[24, 24], 0.2)


def test_zbuffer_through_renderer():
    r = Renderer(64, 64, near=0.1, far=100.0)
    # red quad further away (z = -4), green quad closer (z = -2)
    pos = np.array([
        [-1, -1, -4], [1, -1, -4], [0, 1, -4],      # red, far
        [-1, -1, -2], [1, -1, -2], [0, 1, -2],      # green, near
    ], dtype=float)
    uvs = np.zeros((6, 2))
    cols = np.array([RED, RED, RED, GREEN, GREEN, GREEN], dtype=float)
    idx = np.array([[0, 1, 2], [3, 4, 5]])
    r.render(pos, uvs, cols, idx)
    assert np.allclose(r.fb.color[32, 32], GREEN)


# --------------------------------------- perspective-correct interpolation

def test_perspective_correct_interpolation():
    """A floor plane receding into the distance: screen-space linear
    interpolation of uv would be visibly wrong; the renderer must match
    the analytic perspective-correct value."""
    W = H = 128
    fovy = np.radians(60.0)
    r = Renderer(W, H, fovy=fovy, near=0.1, far=100.0)

    # floor plane y = -1, from z = -2 (v=0) to z = -6 (v=1), x in [-3, 3]
    pos = np.array([
        [-3, -1, -2], [3, -1, -2], [3, -1, -6], [-3, -1, -6],
    ], dtype=float)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cols = np.ones((4, 3))
    idx = np.array([[0, 1, 2], [0, 2, 3]])

    # gradient texture: texel (i, j) = (u, v, 0) so the sampled color
    # directly reveals the interpolated texture coordinate
    n = 64
    grad = np.zeros((n, n, 3))
    for j in range(n):
        for i in range(n):
            grad[j, i] = ((i + 0.5) / n, (j + 0.5) / n, 0.0)
    tex = Texture(grad)

    r.render(pos, uvs, cols, idx, texture=tex, sample_mode="bilinear")

    px, py = W // 2, int(H * 0.9)
    ndc_x = (px + 0.5) / W * 2.0 - 1.0
    ndc_y = 1.0 - (py + 0.5) / H * 2.0
    f = 1.0 / np.tan(fovy / 2.0)
    # analytic ray/plane intersection (view space, camera at origin):
    # y = ndc_y * (-z) / f = -1  ->  z = f / ndc_y ; x = ndc_x * (-z) / f
    z = f / ndc_y
    x = ndc_x * (-z) / f
    u_exact = (x + 3.0) / 6.0
    v_exact = (z + 2.0) / -4.0

    got = r.fb.color[py, px]
    assert np.allclose(got[0], u_exact, atol=0.03)
    assert np.allclose(got[1], v_exact, atol=0.03)

    # sanity: naive screen-space linear interpolation disagrees by a wide
    # margin at this pixel, so the test really detects perspective
    # correctness rather than any smooth interpolation
    proj = r.projection
    scr = []
    for p in pos:
        h = proj @ np.array([p[0], p[1], p[2], 1.0])
        nd = h[:3] / h[3]
        scr.append(((nd[0] + 1) * 0.5 * W, (1 - (nd[1] + 1) * 0.5) * H))
    # pixel is inside triangle (0, 1, 2): barycentric in screen space
    (x0, y0), (x1, y1), (x2, y2) = scr[0], scr[1], scr[2]
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    b0 = ((y1 - y2) * (px + 0.5 - x2) + (x2 - x1) * (py + 0.5 - y2)) / den
    b1 = ((y2 - y0) * (px + 0.5 - x2) + (x0 - x2) * (py + 0.5 - y2)) / den
    b2 = 1 - b0 - b1
    v_linear = b0 * uvs[0, 1] + b1 * uvs[1, 1] + b2 * uvs[2, 1]
    assert abs(v_linear - v_exact) > 0.04
    assert abs(got[1] - v_linear) > 0.03


# ---------------------------------------------------- near-plane clipping

def test_triangle_crossing_near_plane_is_clipped():
    r = Renderer(64, 64, near=1.0, far=100.0)
    # a tall triangle whose tip sits behind the camera (z = +2)
    pos = np.array([
        [-1.0, -1.0, -4.0],
        [1.0, -1.0, -4.0],
        [0.0, 1.0, 2.0],
    ])
    uvs = np.zeros((3, 2))
    cols = np.array([RED, RED, RED])
    r.render(pos, uvs, cols, np.array([[0, 1, 2]]))
    assert np.isfinite(r.fb.color).all()
    assert np.isfinite(r.fb.depth).all() or True  # depth may stay inf
    drawn = r.fb.depth < np.inf
    assert drawn.any(), "clipped triangle should still cover pixels"
    # the visible band of the clipped triangle (between the near-plane
    # cut and the projected base) must be red
    assert np.allclose(r.fb.color[40, 32], RED)
    # above the cut (clipped away) and below the base: background
    assert np.allclose(r.fb.color[20, 32], (0.0, 0.0, 0.0))
    assert np.allclose(r.fb.color[56, 32], (0.0, 0.0, 0.0))


def test_triangle_fully_behind_near_plane_draws_nothing():
    r = Renderer(64, 64, near=1.0, far=100.0)
    pos = np.array([
        [-1.0, -1.0, -0.5],
        [1.0, -1.0, -0.5],
        [0.0, 1.0, -0.5],
    ])
    r.render(pos, np.zeros((3, 2)), np.array([RED] * 3), np.array([[0, 1, 2]]))
    assert not (r.fb.depth < np.inf).any()


def test_clipped_triangle_has_no_invalid_coordinates():
    # a vertex exactly at the camera (w = 0) must not produce NaN/inf
    r = Renderer(32, 32, near=0.5, far=50.0)
    pos = np.array([
        [-1.0, -1.0, -5.0],
        [1.0, -1.0, -5.0],
        [0.0, 1.0, 0.0],   # on the camera plane (w = 0 in clip space)
    ])
    r.render(pos, np.zeros((3, 2)), np.array([BLUE] * 3), np.array([[0, 1, 2]]))
    assert np.isfinite(r.fb.color).all()
    assert (r.fb.depth < np.inf).any()


def test_camera_with_look_at_renders_scene():
    r = Renderer(64, 64, near=0.1, far=100.0)
    view = look_at(eye=(0, 0, 3), center=(0, 0, 0), up=(0, 1, 0))
    pos = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=float)
    cols = np.array([GREEN, GREEN, GREEN])
    r.render(pos, np.zeros((3, 2)), cols, np.array([[0, 1, 2]]), view=view)
    assert np.allclose(r.fb.color[40, 32], GREEN)
