"""Texture sampling tests: nearest, bilinear, clamp and repeat borders."""

import numpy as np

from softrenderer import Texture


def make_2x2():
    # top-left red, top-right green, bottom-left blue, bottom-right white
    data = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    ])
    return Texture(data)


def test_nearest_picks_exact_texel():
    tex = make_2x2()
    assert np.allclose(tex.sample(0.25, 0.25, mode="nearest"), [1, 0, 0])
    assert np.allclose(tex.sample(0.75, 0.25, mode="nearest"), [0, 1, 0])
    assert np.allclose(tex.sample(0.25, 0.75, mode="nearest"), [0, 0, 1])
    assert np.allclose(tex.sample(0.75, 0.75, mode="nearest"), [1, 1, 1])


def test_bilinear_center_is_average():
    tex = make_2x2()
    c = tex.sample(0.5, 0.5, mode="bilinear")
    assert np.allclose(c, [0.5, 0.5, 0.5])


def test_bilinear_halfway_horizontal():
    tex = make_2x2()
    # halfway between the two top texel centers
    c = tex.sample(0.5, 0.25, mode="bilinear")
    assert np.allclose(c, [0.5, 0.5, 0.0])


def test_bilinear_at_texel_center_returns_texel():
    tex = make_2x2()
    assert np.allclose(tex.sample(0.25, 0.25, mode="bilinear"), [1, 0, 0])
    assert np.allclose(tex.sample(0.75, 0.75, mode="bilinear"), [1, 1, 1])


def test_clamp_border():
    tex = make_2x2()
    # outside [0, 1] clamps to the edge texels
    assert np.allclose(tex.sample(-0.5, 0.25, mode="nearest", wrap="clamp"),
                       [1, 0, 0])
    assert np.allclose(tex.sample(1.5, 0.25, mode="nearest", wrap="clamp"),
                       [0, 1, 0])
    assert np.allclose(tex.sample(0.25, 2.0, mode="nearest", wrap="clamp"),
                       [0, 0, 1])
    # bilinear at a clamped corner equals the corner texel
    assert np.allclose(tex.sample(-1.0, -1.0, mode="bilinear", wrap="clamp"),
                       [1, 0, 0])


def test_repeat_border():
    tex = make_2x2()
    # u = 1.25 wraps to u = 0.25
    assert np.allclose(tex.sample(1.25, 0.25, mode="nearest", wrap="repeat"),
                       [1, 0, 0])
    # u = -0.25 wraps to 0.75, v = 0.75 -> bottom-right texel (white)
    assert np.allclose(tex.sample(-0.25, 0.75, mode="nearest", wrap="repeat"),
                       [1, 1, 1])


def test_checker_texture():
    tex = Texture.checker(8, 8, 2, 2, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert np.allclose(tex.sample(0.125, 0.125, mode="nearest"), [0, 0, 0])
    assert np.allclose(tex.sample(0.625, 0.125, mode="nearest"), [1, 1, 1])
    assert np.allclose(tex.sample(0.625, 0.625, mode="nearest"), [0, 0, 0])
