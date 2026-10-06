"""Matrix transform tests: model / view / projection / viewport chain."""
import numpy as np

from softrender import math3d


def test_translate_scale_rotate():
    m = math3d.translate(1, 2, 3) @ math3d.scale(2, 2, 2)
    p = m @ np.array([1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(p[:3], [3.0, 4.0, 5.0])

    r = math3d.rotate_z(np.pi / 2)
    p = r @ np.array([1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(p[:3], [0.0, 1.0, 0.0], atol=1e-12)


def test_look_at_moves_origin_to_minus_distance():
    view = math3d.look_at(eye=(0, 0, 5), center=(0, 0, 0), up=(0, 1, 0))
    p = view @ np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(p[:3], [0.0, 0.0, -5.0], atol=1e-12)
    # Camera x axis maps world +x to view +x.
    p = view @ np.array([1.0, 0.0, -5.0, 1.0])
    np.testing.assert_allclose(p[:3], [1.0, 0.0, -10.0], atol=1e-12)


def test_perspective_known_point():
    # fovy=90deg, aspect=1 -> f = 1/tan(45deg) = 1
    proj = math3d.perspective(np.pi / 2, 1.0, 1.0, 10.0)
    p = proj @ np.array([0.0, 0.0, -2.0, 1.0])
    # clip = (0, 0, (11/-9)*(-2) + 20/-9, 2) = (0, 0, 2/9, 2)
    np.testing.assert_allclose(p, [0.0, 0.0, 2.0 / 9.0, 2.0], atol=1e-12)
    ndc = p[:3] / p[3]
    np.testing.assert_allclose(ndc, [0.0, 0.0, 1.0 / 9.0], atol=1e-12)
    # Near plane maps to ndc z = -1, far plane to +1.
    for z_view, z_ndc in [(-1.0, -1.0), (-10.0, 1.0)]:
        q = proj @ np.array([0.0, 0.0, z_view, 1.0])
        np.testing.assert_allclose(q[2] / q[3], z_ndc, atol=1e-12)


def test_full_mvp_viewport_chain():
    model = math3d.translate(0, 0, -3)
    view = math3d.look_at((0, 0, 0), (0, 0, -1), (0, 1, 0))
    proj = math3d.perspective(np.pi / 2, 1.0, 1.0, 100.0)
    vp = math3d.viewport(0, 0, 64, 64)
    mvp = proj @ view @ model
    clip = mvp @ np.array([0.0, 0.0, 0.0, 1.0])
    ndc = clip[:3] / clip[3]
    np.testing.assert_allclose(ndc[:2], [0.0, 0.0], atol=1e-12)
    screen = vp @ np.append(ndc, 1.0)
    # World origin lands at the center of a 64x64 viewport.
    np.testing.assert_allclose(screen[:2], [32.0, 32.0], atol=1e-12)
    # Off-center point: view-space (1, 0, -3) -> ndc x = 1/3.
    clip = mvp @ np.array([1.0, 0.0, -3.0 + 3.0, 1.0])  # model shifts z by -3
    clip = proj @ view @ np.array([1.0, 0.0, -3.0, 1.0])
    ndc = clip[:3] / clip[3]
    np.testing.assert_allclose(ndc[0], 1.0 / 3.0, atol=1e-12)
    screen = vp @ np.append(ndc, 1.0)
    np.testing.assert_allclose(screen[0], (1.0 / 3.0 + 1.0) * 32.0, atol=1e-12)
