"""Matrix / transform tests: model, view, projection and the full MVP chain."""

import numpy as np

from softrenderer import (
    identity,
    look_at,
    perspective,
    rotate_z,
    scale,
    transform_point,
    translate,
)


def test_translate_moves_point():
    m = translate(1.0, 2.0, 3.0)
    h = transform_point(m, (0.0, 0.0, 0.0))
    assert np.allclose(h, [1.0, 2.0, 3.0, 1.0])


def test_scale_scales_point():
    m = scale(2.0, 3.0, 4.0)
    h = transform_point(m, (1.0, 1.0, 1.0))
    assert np.allclose(h, [2.0, 3.0, 4.0, 1.0])


def test_rotate_z_90_degrees():
    m = rotate_z(np.pi / 2.0)
    h = transform_point(m, (1.0, 0.0, 0.0))
    assert np.allclose(h, [0.0, 1.0, 0.0, 1.0], atol=1e-12)


def test_model_chain_order():
    # scale first, then translate (column-vector convention: T @ S @ p)
    m = translate(10.0, 0.0, 0.0) @ scale(2.0, 2.0, 2.0)
    h = transform_point(m, (1.0, 0.0, 0.0))
    assert np.allclose(h, [12.0, 0.0, 0.0, 1.0])


def test_look_at_identity_when_looking_down_minus_z():
    m = look_at(eye=(0, 0, 0), center=(0, 0, -1), up=(0, 1, 0))
    assert np.allclose(m, identity(), atol=1e-12)


def test_look_at_puts_center_on_minus_z_axis():
    eye = np.array([1.0, 2.0, 3.0])
    center = np.array([0.0, 0.0, 0.0])
    m = look_at(eye=eye, center=center, up=(0, 1, 0))
    h = transform_point(m, center)
    dist = np.linalg.norm(center - eye)
    assert np.allclose(h[:2], [0.0, 0.0], atol=1e-12)
    assert np.isclose(h[2], -dist)  # center ends up on -Z at view distance


def test_perspective_maps_near_and_far_to_ndc():
    near, far = 0.5, 10.0
    p = perspective(np.radians(60.0), aspect=1.0, near=near, far=far)
    hn = transform_point(p, (0.0, 0.0, -near))
    hf = transform_point(p, (0.0, 0.0, -far))
    assert np.isclose(hn[2] / hn[3], -1.0)
    assert np.isclose(hf[2] / hf[3], 1.0)


def test_perspective_center_axis_stays_centered():
    p = perspective(np.radians(90.0), aspect=2.0, near=0.1, far=100.0)
    h = transform_point(p, (0.0, 0.0, -5.0))
    assert np.isclose(h[0] / h[3], 0.0)
    assert np.isclose(h[1] / h[3], 0.0)


def test_perspective_fov():
    # fovy = 90 deg, aspect 1: a point at 45 deg from the axis lands on
    # the NDC edge x = 1.
    p = perspective(np.radians(90.0), aspect=1.0, near=0.1, far=100.0)
    h = transform_point(p, (5.0, 0.0, -5.0))
    assert np.isclose(h[0] / h[3], 1.0)


def test_full_mvp_pipeline_to_screen():
    # Camera at z = 3 looking at the origin; a model point at the origin
    # must land on the screen center.
    from softrenderer import Renderer

    r = Renderer(64, 64, fovy=np.radians(60.0), near=0.1, far=100.0)
    model = identity()
    view = look_at(eye=(0, 0, 3), center=(0, 0, 0), up=(0, 1, 0))
    mvp = r.projection @ view @ model
    h = transform_point(mvp, (0.0, 0.0, 0.0))
    ndc = h[:3] / h[3]
    sx = (ndc[0] + 1.0) * 0.5 * 64
    sy = (1.0 - (ndc[1] + 1.0) * 0.5) * 64
    assert np.isclose(sx, 32.0)
    assert np.isclose(sy, 32.0)
    assert -1.0 < ndc[2] < 1.0  # inside the depth range
