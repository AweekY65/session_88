"""Pure-CPU software rasterizer.

Pipeline per triangle:
  1. Vertex transform: clip = P @ V @ M @ [x, y, z, 1]^T
  2. Near-plane clipping in homogeneous clip space (keep z >= -w)
  3. Perspective divide -> NDC, viewport transform -> screen space
  4. Rasterization with barycentric coordinates (pixel-center sampling)
  5. Z-buffer depth test (smaller depth wins)
  6. Perspective-correct attribute interpolation (1/w weighting)
  7. Optional texture sampling (nearest / bilinear) modulated by vertex color
"""
import numpy as np

from .math3d import identity, viewport as viewport_matrix
from .texture import Texture


class Vertex:
    """A vertex carrying clip-space position and attributes."""

    __slots__ = ("clip", "uv", "color")

    def __init__(self, clip, uv, color):
        self.clip = np.asarray(clip, dtype=np.float64)  # vec4, homogeneous
        self.uv = np.asarray(uv, dtype=np.float64)      # vec2
        self.color = np.asarray(color, dtype=np.float64)  # vec3

    def lerp(self, other, t):
        return Vertex(
            self.clip + (other.clip - self.clip) * t,
            self.uv + (other.uv - self.uv) * t,
            self.color + (other.color - self.color) * t,
        )


def clip_triangle_near(verts):
    """Sutherland-Hodgman clip of one triangle against the near plane.

    Clip-space half-space kept: z >= -w  (i.e. z + w >= 0), which is the
    camera-space half-space z <= -near. Returns a list of 0..2 triangles
    (lists of 3 Vertex). Linear interpolation in clip space is exact here
    because clipping happens before the perspective divide.
    """
    def depth(v):
        return v.clip[2] + v.clip[3]

    polygon = list(verts)
    clipped = []
    n = len(polygon)
    for i in range(n):
        cur = polygon[i]
        prev = polygon[(i - 1) % n]
        d_cur = depth(cur)
        d_prev = depth(prev)
        cur_in = d_cur >= 0.0
        prev_in = d_prev >= 0.0
        if cur_in != prev_in:
            t = d_prev / (d_prev - d_cur)
            clipped.append(prev.lerp(cur, t))
        if cur_in:
            clipped.append(cur)
    tris = []
    for i in range(1, len(clipped) - 1):
        tris.append([clipped[0], clipped[i], clipped[i + 1]])
    return tris


def barycentric(px, py, a, b, c):
    """Barycentric coordinates of point (px, py) w.r.t. screen triangle a,b,c.

    Returns (w0, w1, w2) with w0 + w1 + w2 = 1, or None if degenerate.
    """
    d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(d) < 1e-12:
        return None
    w0 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / d
    w1 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / d
    w2 = 1.0 - w0 - w1
    return w0, w1, w2


class Framebuffer:
    def __init__(self, width, height, clear_color=(0.0, 0.0, 0.0)):
        self.width = width
        self.height = height
        self.color = np.zeros((height, width, 3), dtype=np.float64)
        self.color[:, :] = np.asarray(clear_color, dtype=np.float64)
        self.depth = np.full((height, width), np.inf, dtype=np.float64)

    def clear(self, color=(0.0, 0.0, 0.0)):
        self.color[:, :] = np.asarray(color, dtype=np.float64)
        self.depth[:, :] = np.inf


class Rasterizer:
    """Renders indexed triangle meshes into a Framebuffer."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.framebuffer = Framebuffer(width, height)

    def draw(self, positions, indices, uvs=None, colors=None,
             model=None, view=None, projection=None,
             texture=None, texture_mode="bilinear"):
        """Render triangles.

        positions: (N, 3) model-space positions.
        indices:   (M, 3) triangle vertex indices.
        uvs:       (N, 2) texture coordinates (default zeros).
        colors:    (N, 3) vertex colors in [0, 1] (default white).
        model/view/projection: 4x4 matrices (default identity).
        texture:   optional Texture; final color = vertex color * texel.
        """
        positions = np.asarray(positions, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        n = positions.shape[0]
        if uvs is None:
            uvs = np.zeros((n, 2))
        if colors is None:
            colors = np.ones((n, 3))
        uvs = np.asarray(uvs, dtype=np.float64)
        colors = np.asarray(colors, dtype=np.float64)
        model = identity() if model is None else np.asarray(model, dtype=np.float64)
        view = identity() if view is None else np.asarray(view, dtype=np.float64)
        projection = identity() if projection is None else np.asarray(
            projection, dtype=np.float64)
        mvp = projection @ view @ model
        vp = viewport_matrix(0, 0, self.width, self.height)

        for tri in indices:
            verts = []
            for idx in tri:
                p = np.append(positions[idx], 1.0)
                verts.append(Vertex(mvp @ p, uvs[idx], colors[idx]))
            for clipped in clip_triangle_near(verts):
                self._rasterize(clipped, vp, texture, texture_mode)

    def _rasterize(self, verts, vp, texture, texture_mode):
        fb = self.framebuffer
        # Perspective divide -> NDC, then viewport -> screen.
        screen = []
        inv_w = []
        for v in verts:
            w = v.clip[3]
            if w <= 0.0:
                return  # behind the eye after clipping; discard defensively
            ndc = v.clip[:3] / w
            sp = vp @ np.array([ndc[0], ndc[1], ndc[2], 1.0])
            screen.append(sp[:3])
            inv_w.append(1.0 / w)

        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = screen
        min_x = max(int(np.floor(min(x0, x1, x2))), 0)
        max_x = min(int(np.ceil(max(x0, x1, x2))), self.width - 1)
        min_y = max(int(np.floor(min(y0, y1, y2))), 0)
        max_y = min(int(np.ceil(max(y0, y1, y2))), self.height - 1)
        if min_x > max_x or min_y > max_y:
            return

        a = (x0, y0)
        b = (x1, y1)
        c = (x2, y2)
        eps = 1e-9
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                bc = barycentric(px + 0.5, py + 0.5, a, b, c)
                if bc is None:
                    continue
                w0, w1, w2 = bc
                if w0 < -eps or w1 < -eps or w2 < -eps:
                    continue
                # NDC z is linear in screen space -> plain interpolation.
                depth = w0 * z0 + w1 * z1 + w2 * z2
                if depth < -1.0 - 1e-9 or depth > 1.0 + 1e-9:
                    continue
                if depth >= fb.depth[py, px]:
                    continue
                # Perspective-correct interpolation: weight by 1/w.
                iw = w0 * inv_w[0] + w1 * inv_w[1] + w2 * inv_w[2]
                if iw <= 0.0:
                    continue
                k0 = w0 * inv_w[0] / iw
                k1 = w1 * inv_w[1] / iw
                k2 = w2 * inv_w[2] / iw
                uv = (k0 * verts[0].uv + k1 * verts[1].uv + k2 * verts[2].uv)
                color = (k0 * verts[0].color + k1 * verts[1].color
                         + k2 * verts[2].color)
                if texture is not None:
                    color = color * texture.sample(uv[0], uv[1], texture_mode)
                fb.depth[py, px] = depth
                fb.color[py, px] = np.clip(color, 0.0, 1.0)


def render(width, height, positions, indices, **kwargs):
    """Convenience one-shot render; returns the Rasterizer."""
    r = Rasterizer(width, height)
    r.draw(positions, indices, **kwargs)
    return r
