"""Near-plane clipping tests (homogeneous clip space, plane z = -w)."""

import numpy as np

from softrenderer import clip_triangle_near, perspective


def make_vert(x, y, z_view, proj, uv=(0.0, 0.0), color=(1.0, 1.0, 1.0)):
    """Build a clip-space vertex from a view-space position."""
    pos = proj @ np.array([x, y, z_view, 1.0])
    return (pos, np.asarray(uv, dtype=np.float64),
            np.asarray(color, dtype=np.float64))


PROJ = perspective(np.radians(60.0), aspect=1.0, near=0.5, far=50.0)


def test_fully_inside_triangle_is_untouched():
    tri = [make_vert(-1, -1, -5, PROJ), make_vert(1, -1, -5, PROJ),
           make_vert(0, 1, -5, PROJ)]
    out = clip_triangle_near(tri)
    assert len(out) == 1
    for orig, kept in zip(tri, out[0]):
        assert np.allclose(orig[0], kept[0])


def test_fully_behind_near_plane_produces_nothing():
    tri = [make_vert(-1, -1, -0.1, PROJ), make_vert(1, -1, -0.1, PROJ),
           make_vert(0, 1, -0.1, PROJ)]
    assert clip_triangle_near(tri) == []


def test_fully_behind_camera_produces_nothing():
    tri = [make_vert(-1, -1, 2.0, PROJ), make_vert(1, -1, 2.0, PROJ),
           make_vert(0, 1, 2.0, PROJ)]
    assert clip_triangle_near(tri) == []


def test_one_vertex_inside_gives_one_triangle():
    tri = [make_vert(0, 0, -5, PROJ), make_vert(-1, 0, 1.0, PROJ),
           make_vert(1, 0, 1.0, PROJ)]
    out = clip_triangle_near(tri)
    assert len(out) == 1
    # every generated vertex must satisfy the near-plane inequality
    for t in out:
        for pos, _, _ in t:
            assert pos[3] + pos[2] >= -1e-9


def test_two_vertices_inside_gives_two_triangles():
    tri = [make_vert(-0.5, 0, -5, PROJ), make_vert(0.5, 0, -5, PROJ),
           make_vert(0, 1, 1.0, PROJ)]
    out = clip_triangle_near(tri)
    assert len(out) == 2
    for t in out:
        for pos, _, _ in t:
            assert pos[3] + pos[2] >= -1e-9


def test_intersection_lies_on_near_plane_and_interpolates_attributes():
    uv0, uv1 = (0.0, 0.0), (1.0, 1.0)
    a = make_vert(0, 0, -1.0, PROJ, uv=uv0)   # in front of the near plane
    b = make_vert(0, 0, 1.5, PROJ, uv=uv1)    # behind the camera
    c = make_vert(0.5, 0, -1.0, PROJ, uv=uv0)
    out = clip_triangle_near([a, b, c])
    assert len(out) >= 1
    # find a generated vertex that is not an original one: it must lie
    # exactly on the plane (w + z == 0)
    on_plane = [v for t in out for v in t
                if abs(v[0][3] + v[0][2]) < 1e-9]
    assert on_plane, "expected a clipped vertex on the near plane"
    # attribute interpolation in clip space: the intersection on edge a-b
    # is at t = d_a / (d_a - d_b) with d = w + z.
    da = a[0][3] + a[0][2]
    db = b[0][3] + b[0][2]
    t = da / (da - db)
    expected_uv = np.array(uv0) + t * (np.array(uv1) - np.array(uv0))
    found = any(np.allclose(v[1], expected_uv, atol=1e-9)
                for tt in out for v in tt)
    assert found, "uv was not interpolated correctly at the clip point"
