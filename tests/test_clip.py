"""Near-plane clipping tests (clip space + render level)."""
import numpy as np

from softrender import Rasterizer, Vertex, clip_triangle_near, math3d


def _v(x, y, z, w=1.0):
    return Vertex([x, y, z, w], [0, 0], [1, 1, 1])


def test_fully_inside_passthrough():
    tris = clip_triangle_near([_v(0, 0, 0), _v(1, 0, 0), _v(0, 1, 0)])
    assert len(tris) == 1
    np.testing.assert_allclose(tris[0][1].clip, [1, 0, 0, 1])


def test_fully_behind_near_plane_discarded():
    # z + w < 0 for all vertices (e.g. between camera and near plane).
    tris = clip_triangle_near([_v(0, 0, -2), _v(1, 0, -2), _v(0, 1, -2)])
    assert tris == []


def test_one_vertex_behind_produces_two_triangles():
    tris = clip_triangle_near([_v(0, 0, 1), _v(1, 0, 1), _v(0, 1, -2)])
    assert len(tris) == 2
    for tri in tris:
        for v in tri:
            assert v.clip[2] + v.clip[3] >= -1e-12
    # The two new vertices lie exactly on the near plane z + w = 0.
    on_plane = [v for tri in tris for v in tri
                if abs(v.clip[2] + v.clip[3]) < 1e-9]
    assert len(on_plane) >= 2


def test_two_vertices_behind_produce_one_triangle():
    tris = clip_triangle_near([_v(0, 0, 1), _v(1, 0, -2), _v(0, 1, -2)])
    assert len(tris) == 1
    for v in tris[0]:
        assert v.clip[2] + v.clip[3] >= -1e-12


def test_attributes_interpolated_at_clip_boundary():
    a = Vertex([0, 0, 1, 1], [0.0, 0.0], [1.0, 0.0, 0.0])
    b = Vertex([1, 0, -3, 1], [1.0, 0.0], [0.0, 1.0, 0.0])
    c = Vertex([0, 1, 1, 1], [0.0, 1.0], [0.0, 0.0, 1.0])
    tris = clip_triangle_near([a, b, c])
    # Edge a->b crosses z+w=0 at t = 2/(2+2) = 0.5.
    boundary = [v for tri in tris for v in tri
                if abs(v.clip[2] + v.clip[3]) < 1e-9]
    np.testing.assert_allclose(boundary[0].uv, [0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(boundary[0].color, [0.5, 0.5, 0.0], atol=1e-12)


def test_render_triangle_crossing_near_plane():
    # Camera at origin looking down -z, near = 1. One vertex sits between
    # camera and near plane (z = -0.5); the triangle must be clipped, not
    # produce NaN or garbage coordinates.
    r = Rasterizer(64, 64)
    proj = math3d.perspective(np.pi / 2, 1.0, 1.0, 100.0)
    r.draw(
        positions=[[0.0, 0.0, -0.5], [-3.0, -3.0, -5.0], [3.0, -3.0, -5.0]],
        indices=[[0, 1, 2]],
        colors=[[1, 0, 0]] * 3,
        projection=proj,
    )
    fb = r.framebuffer
    assert np.isfinite(fb.color).all()
    covered = np.isfinite(fb.depth)
    assert covered.sum() > 100  # clipped triangle is still visible
    assert (fb.depth[covered] >= -1.0 - 1e-9).all()
    assert (fb.depth[covered] <= 1.0 + 1e-9).all()
    # Symmetric triangle -> symmetric coverage about the vertical axis.
    np.testing.assert_array_equal(covered, covered[:, ::-1])


def test_render_triangle_fully_behind_near_plane_draws_nothing():
    r = Rasterizer(32, 32)
    proj = math3d.perspective(np.pi / 2, 1.0, 1.0, 100.0)
    r.draw(
        positions=[[-1, -1, -0.5], [1, -1, -0.5], [0, 1, -0.5]],
        indices=[[0, 1, 2]],
        projection=proj,
    )
    assert np.isinf(r.framebuffer.depth).all()
