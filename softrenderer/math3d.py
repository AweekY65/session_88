"""3D math: 4x4 matrices, camera and projection helpers.

Conventions
-----------
* Column vectors: transformed point is ``M @ [x, y, z, 1]``.
* Right-handed coordinate system. In view space the camera sits at the
  origin and looks down the -Z axis, +Y is up, +X is right.
* The perspective projection maps view space to OpenGL-style clip space,
  i.e. after division by ``w`` the visible volume is the cube
  ``x, y, z in [-1, 1]`` with ``z = -1`` at the near plane and ``z = +1``
  at the far plane.
"""

import numpy as np


def identity():
    return np.eye(4, dtype=np.float64)


def translate(tx, ty, tz):
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def scale(sx, sy, sz):
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def rotate_x(radians):
    c, s = np.cos(radians), np.sin(radians)
    m = np.eye(4, dtype=np.float64)
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def rotate_y(radians):
    c, s = np.cos(radians), np.sin(radians)
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def rotate_z(radians):
    c, s = np.cos(radians), np.sin(radians)
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def look_at(eye, center, up):
    """View matrix moving the world so `eye` sits at the origin looking
    at `center` (down -Z in view space)."""
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def perspective(fovy_radians, aspect, near, far):
    """OpenGL-style perspective projection (NDC z in [-1, 1])."""
    if near <= 0.0:
        raise ValueError("near plane distance must be positive")
    if far <= near:
        raise ValueError("far plane must be beyond the near plane")
    f = 1.0 / np.tan(fovy_radians / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def transform_point(m, point):
    """Apply 4x4 matrix `m` to a 3D point, returning the homogeneous 4-vector."""
    p = np.asarray(point, dtype=np.float64)
    return m @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
