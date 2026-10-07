"""Homogeneous (clip-space) triangle clipping against the near plane.

A clip-space vertex ``(x, y, z, w)`` is in front of the near plane when
``z >= -w`` (OpenGL convention). Clipping is done with the
Sutherland-Hodgman algorithm restricted to this single plane, before the
perspective divide, so vertices behind the camera (``w <= 0``) never
produce invalid screen coordinates. Vertex attributes (uv, color) are
interpolated linearly in clip space at the intersection point, which is
the correct thing to do because perspective-correct interpolation later
only needs attribute values that are linear in clip space.
"""

import numpy as np


def _near_distance(clip_pos):
    """Signed distance-like value: >= 0 means in front of the near plane."""
    return clip_pos[3] + clip_pos[2]  # w + z >= 0  <=>  z >= -w


def clip_triangle_near(verts):
    """Clip one triangle against the near plane.

    Parameters
    ----------
    verts : list of 3 tuples ``(clip_pos, uv, color)``
        ``clip_pos`` is a length-4 array, ``uv`` length-2, ``color`` length-3.

    Returns
    -------
    list of triangles (each a list of 3 ``(clip_pos, uv, color)`` tuples).
    Empty when the triangle lies entirely behind the near plane;
    one or two triangles otherwise.
    """
    out = []
    n = len(verts)
    for i in range(n):
        cur = verts[i]
        nxt = verts[(i + 1) % n]
        d_cur = _near_distance(cur[0])
        d_nxt = _near_distance(nxt[0])
        cur_in = d_cur >= 0.0
        nxt_in = d_nxt >= 0.0
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            t = d_cur / (d_cur - d_nxt)
            pos = cur[0] + t * (nxt[0] - cur[0])
            uv = cur[1] + t * (nxt[1] - cur[1])
            color = cur[2] + t * (nxt[2] - cur[2])
            out.append((pos, uv, color))
    if len(out) < 3:
        return []
    # Fan-triangulate the resulting convex polygon (3 or 4 vertices).
    tris = []
    for i in range(1, len(out) - 1):
        tris.append([out[0], out[i], out[i + 1]])
    return tris
