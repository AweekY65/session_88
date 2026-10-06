"""Shared deterministic test scene: a perspective checker floor plus two
overlapping colored triangles (near one must win the depth test)."""
import numpy as np

from softrender import Rasterizer, Texture, math3d


def checker_texture(size=8, cells=4):
    data = np.zeros((size, size, 3))
    for y in range(size):
        for x in range(size):
            on = ((x * cells // size) + (y * cells // size)) % 2 == 0
            data[y, x] = (1.0, 1.0, 1.0) if on else (0.1, 0.1, 0.1)
    return Texture(data, wrap="repeat")


def render_scene(width=64, height=64):
    r = Rasterizer(width, height)
    view = math3d.look_at((0, 1.5, 3.0), (0, 0.0, -2.0), (0, 1, 0))
    proj = math3d.perspective(np.pi / 3, width / height, 0.5, 50.0)

    # Textured floor quad receding toward the horizon.
    r.draw(
        positions=[[-6, 0, 2], [6, 0, 2], [6, 0, -30], [-6, 0, -30]],
        indices=[[0, 1, 2], [0, 2, 3]],
        uvs=[[0, 0], [6, 0], [6, 32], [0, 32]],
        texture=checker_texture(),
        texture_mode="bilinear",
        view=view,
        projection=proj,
    )
    # Far red triangle, then overlapping near green triangle.
    r.draw(
        positions=[[-1.2, 0.2, -3.0], [0.4, 0.2, -3.0], [-0.4, 1.4, -3.0]],
        indices=[[0, 1, 2]],
        colors=[[0.9, 0.1, 0.1]] * 3,
        view=view,
        projection=proj,
    )
    r.draw(
        positions=[[-0.6, 0.1, -2.0], [1.0, 0.1, -2.0], [0.2, 1.0, -2.0]],
        indices=[[0, 1, 2]],
        colors=[[0.1, 0.9, 0.2]] * 3,
        view=view,
        projection=proj,
    )
    return r
