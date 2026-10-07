"""Framebuffer, triangle rasterization and the full render pipeline.

Pipeline
--------
model -> view -> projection  (vertex positions, column-vector convention)
      -> near-plane clipping in homogeneous clip space
      -> perspective divide (NDC)
      -> viewport transform (screen space, y flipped so +Y is up)
      -> rasterization with barycentric coordinates
      -> perspective-correct attribute interpolation
      -> z-buffer depth test
      -> texture sampling
"""

import numpy as np

from .clip import clip_triangle_near
from .imageio import write_ppm, write_png
from .math3d import identity, perspective

_EPS = 1e-9


class Framebuffer:
    """Color (float64 RGB in [0, 1]) plus a depth buffer (smaller = nearer)."""

    def __init__(self, width, height, clear_color=(0.0, 0.0, 0.0)):
        self.width = int(width)
        self.height = int(height)
        self.clear_color = tuple(clear_color)
        self.color = np.zeros((self.height, self.width, 3), dtype=np.float64)
        self.depth = np.full((self.height, self.width), np.inf, dtype=np.float64)
        self.clear(clear_color)

    def clear(self, color=None):
        if color is None:
            color = self.clear_color
        self.color[:, :] = color
        self.depth[:, :] = np.inf

    def color_uint8(self):
        return (np.clip(self.color, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    def save_ppm(self, path):
        write_ppm(path, self.color_uint8())

    def save_png(self, path):
        write_png(path, self.color_uint8())


def _edge(p0, p1, px, py):
    """Edge function: twice the signed area of triangle (p0, p1, (px, py))."""
    return (px - p0[0]) * (p1[1] - p0[1]) - (py - p0[1]) * (p1[0] - p0[0])


def rasterize_triangle(fb, pts, uvs, colors, inv_w, texture=None,
                       sample_mode="bilinear", wrap="clamp"):
    """Rasterize one triangle in screen space.

    Parameters
    ----------
    fb : Framebuffer
    pts : (3, 3) array
        Screen-space vertices ``(x, y, z)``. Pixel centers are at half
        integers; ``z`` is the depth value (smaller wins).
    uvs : (3, 2) array
    colors : (3, 3) array
    inv_w : (3,) array
        ``1 / w_clip`` per vertex, used for perspective-correct
        interpolation of ``uvs`` and ``colors``.
    texture : Texture or None
    """
    pts = np.asarray(pts, dtype=np.float64)
    uvs = np.asarray(uvs, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    inv_w = np.asarray(inv_w, dtype=np.float64)

    v0, v1, v2 = pts[0, :2], pts[1, :2], pts[2, :2]
    area = _edge(v1, v2, v0[0], v0[1])
    if abs(area) < _EPS:
        return  # degenerate (zero-area) triangle

    minx = max(int(np.floor(pts[:, 0].min() - 0.5)), 0)
    maxx = min(int(np.ceil(pts[:, 0].max() - 0.5)), fb.width - 1)
    miny = max(int(np.floor(pts[:, 1].min() - 0.5)), 0)
    maxy = min(int(np.ceil(pts[:, 1].max() - 0.5)), fb.height - 1)
    if minx > maxx or miny > maxy:
        return

    for py in range(miny, maxy + 1):
        cy = py + 0.5
        for px in range(minx, maxx + 1):
            cx = px + 0.5
            # Barycentric coordinates; dividing by the same signed area
            # makes the test winding-agnostic.
            w0 = _edge(v1, v2, cx, cy) / area
            w1 = _edge(v2, v0, cx, cy) / area
            w2 = _edge(v0, v1, cx, cy) / area
            if w0 < -_EPS or w1 < -_EPS or w2 < -_EPS:
                continue

            # z/w is linear in screen space, so plain barycentric
            # interpolation of z is correct for the depth test.
            z = w0 * pts[0, 2] + w1 * pts[1, 2] + w2 * pts[2, 2]
            if not z < fb.depth[py, px]:
                continue

            # Perspective-correct interpolation: attributes divided by w
            # are linear in screen space, so interpolate a/w and 1/w,
            # then divide.
            iw = w0 * inv_w[0] + w1 * inv_w[1] + w2 * inv_w[2]
            if abs(iw) < _EPS:
                continue
            u = (w0 * uvs[0, 0] * inv_w[0] + w1 * uvs[1, 0] * inv_w[1]
                 + w2 * uvs[2, 0] * inv_w[2]) / iw
            v = (w0 * uvs[0, 1] * inv_w[0] + w1 * uvs[1, 1] * inv_w[1]
                 + w2 * uvs[2, 1] * inv_w[2]) / iw
            color = ((w0 * colors[0] * inv_w[0] + w1 * colors[1] * inv_w[1]
                      + w2 * colors[2] * inv_w[2]) / iw)

            if texture is not None:
                color = color * texture.sample(u, v, mode=sample_mode, wrap=wrap)

            fb.depth[py, px] = z
            fb.color[py, px] = color


class Renderer:
    """Owns a framebuffer and the projection matrix; draws indexed meshes."""

    def __init__(self, width, height, fovy=np.radians(60.0), aspect=None,
                 near=0.1, far=100.0, clear_color=(0.0, 0.0, 0.0)):
        self.fb = Framebuffer(width, height, clear_color)
        self.near = near
        self.far = far
        if aspect is None:
            aspect = width / float(height)
        self.projection = perspective(fovy, aspect, near, far)

    def render(self, positions, uvs, colors, indices, model=None, view=None,
               texture=None, sample_mode="bilinear", wrap="clamp"):
        """Transform, clip and rasterize an indexed triangle mesh.

        positions : (N, 3) model-space vertices
        uvs       : (N, 2) texture coordinates
        colors    : (N, 3) vertex colors
        indices   : (M, 3) triangle indices
        """
        if model is None:
            model = identity()
        if view is None:
            view = identity()
        positions = np.asarray(positions, dtype=np.float64)
        uvs = np.asarray(uvs, dtype=np.float64)
        colors = np.asarray(colors, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)

        mvp = self.projection @ view @ model
        hom = np.concatenate(
            [positions, np.ones((positions.shape[0], 1))], axis=1)
        clip = (mvp @ hom.T).T  # (N, 4) clip-space positions

        w, h = self.fb.width, self.fb.height
        for tri in indices:
            verts = [(clip[i], uvs[i], colors[i]) for i in tri]
            for clipped in clip_triangle_near(verts):
                pts = np.zeros((3, 3), dtype=np.float64)
                tri_uv = np.zeros((3, 2), dtype=np.float64)
                tri_col = np.zeros((3, 3), dtype=np.float64)
                inv_w = np.zeros(3, dtype=np.float64)
                for k, (pos, uv, col) in enumerate(clipped):
                    ndc = pos[:3] / pos[3]
                    pts[k, 0] = (ndc[0] + 1.0) * 0.5 * w
                    pts[k, 1] = (1.0 - (ndc[1] + 1.0) * 0.5) * h
                    pts[k, 2] = (ndc[2] + 1.0) * 0.5  # 0 near, 1 far
                    tri_uv[k] = uv
                    tri_col[k] = col
                    inv_w[k] = 1.0 / pos[3]
                rasterize_triangle(self.fb, pts, tri_uv, tri_col, inv_w,
                                   texture=texture, sample_mode=sample_mode,
                                   wrap=wrap)
        return self.fb
