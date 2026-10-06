"""4x4 matrix math and transform helpers (row-major numpy, column vectors).

Conventions:
- Right-handed coordinate system.
- Column vectors: transformed point p' = M @ p, with p = [x, y, z, 1]^T.
- Camera looks down its local -Z axis (OpenGL convention).
- Perspective projection maps to NDC in [-1, 1]^3 after division by w.
"""
import numpy as np


def identity():
    return np.eye(4, dtype=np.float64)


def translate(tx, ty, tz):
    m = identity()
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def scale(sx, sy, sz):
    m = identity()
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def rotate_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    m = identity()
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def rotate_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    m = identity()
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def rotate_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    m = identity()
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def look_at(eye, center, up):
    """View matrix transforming world space into camera (view) space."""
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = identity()
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def perspective(fovy, aspect, near, far):
    """OpenGL-style perspective projection matrix (NDC z in [-1, 1])."""
    if near <= 0 or far <= near:
        raise ValueError("require 0 < near < far")
    f = 1.0 / np.tan(fovy / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(left, right, bottom, top, near, far):
    m = identity()
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def viewport(x, y, width, height):
    """Map NDC [-1,1]^2 to pixel coordinates, y axis pointing down."""
    m = identity()
    m[0, 0] = width / 2.0
    m[0, 3] = x + width / 2.0
    m[1, 1] = -height / 2.0
    m[1, 3] = y + height / 2.0
    return m
