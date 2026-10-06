"""Texture sampling tests: nearest / bilinear / boundary handling."""
import numpy as np

from softrender import Rasterizer, Texture

RED = [1.0, 0.0, 0.0]
GREEN = [0.0, 1.0, 0.0]
BLUE = [0.0, 0.0, 1.0]
WHITE = [1.0, 1.0, 1.0]

# 2x2 texture, row 0 is v=0 (top).
TEX = Texture([[RED, GREEN], [BLUE, WHITE]])


def test_nearest_sampling():
    np.testing.assert_allclose(TEX.sample_nearest(0.25, 0.25), RED)
    np.testing.assert_allclose(TEX.sample_nearest(0.75, 0.25), GREEN)
    np.testing.assert_allclose(TEX.sample_nearest(0.25, 0.75), BLUE)
    np.testing.assert_allclose(TEX.sample_nearest(0.75, 0.75), WHITE)


def test_nearest_clamps_out_of_range():
    np.testing.assert_allclose(TEX.sample_nearest(-3.0, 0.25), RED)
    np.testing.assert_allclose(TEX.sample_nearest(4.2, 0.25), GREEN)
    np.testing.assert_allclose(TEX.sample_nearest(0.25, -1.0), RED)
    np.testing.assert_allclose(TEX.sample_nearest(0.25, 9.9), BLUE)


def test_repeat_wrap_mode():
    tex = Texture([[RED, GREEN], [BLUE, WHITE]], wrap="repeat")
    np.testing.assert_allclose(tex.sample_nearest(1.25, 0.25), RED)
    np.testing.assert_allclose(tex.sample_nearest(-0.75, 0.25), RED)
    np.testing.assert_allclose(tex.sample_nearest(2.75, 0.25), GREEN)


def test_bilinear_at_texel_centers():
    np.testing.assert_allclose(TEX.sample_bilinear(0.25, 0.25), RED, atol=1e-12)
    np.testing.assert_allclose(TEX.sample_bilinear(0.75, 0.75), WHITE, atol=1e-12)


def test_bilinear_midpoints():
    np.testing.assert_allclose(TEX.sample_bilinear(0.5, 0.25),
                               (np.array(RED) + np.array(GREEN)) / 2, atol=1e-12)
    np.testing.assert_allclose(TEX.sample_bilinear(0.5, 0.5),
                               (np.array(RED) + np.array(GREEN)
                                + np.array(BLUE) + np.array(WHITE)) / 4,
                               atol=1e-12)


def test_bilinear_clamps_at_border():
    # u=0 is half a texel left of texel 0's center: clamped to the edge texel.
    np.testing.assert_allclose(TEX.sample_bilinear(0.0, 0.25), RED, atol=1e-12)
    np.testing.assert_allclose(TEX.sample_bilinear(-1.0, 0.25), RED, atol=1e-12)
    np.testing.assert_allclose(TEX.sample_bilinear(1.0, 0.75), WHITE, atol=1e-12)


def test_rendered_quad_matches_texture():
    # Axis-aligned quad filling a 4x4 framebuffer, uv spanning [0,1]^2.
    r = Rasterizer(4, 4)
    r.draw(
        positions=[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
        indices=[[0, 1, 2], [0, 2, 3]],
        uvs=[[0, 1], [1, 1], [1, 0], [0, 0]],
        texture=TEX,
        texture_mode="nearest",
    )
    fb = r.framebuffer
    # Screen y=0 row maps to v=0 (top of texture).
    np.testing.assert_allclose(fb.color[0, 0], RED, atol=1e-9)
    np.testing.assert_allclose(fb.color[0, 3], GREEN, atol=1e-9)
    np.testing.assert_allclose(fb.color[3, 0], BLUE, atol=1e-9)
    np.testing.assert_allclose(fb.color[3, 3], WHITE, atol=1e-9)
