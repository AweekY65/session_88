"""Texture storage and sampling (nearest / bilinear, clamp / repeat)."""

import numpy as np


class Texture:
    """An RGB texture backed by a float64 numpy array of shape (H, W, 3).

    Texture coordinates use the usual convention: ``u`` grows to the
    right, ``v`` grows downward, ``(0, 0)`` is the top-left corner and
    ``(1, 1)`` the bottom-right corner of the texture.
    """

    def __init__(self, pixels):
        a = np.asarray(pixels, dtype=np.float64)
        if a.ndim == 2:
            a = np.repeat(a[:, :, None], 3, axis=2)
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError("texture must have shape (H, W) or (H, W, 3)")
        self.data = a
        self.height, self.width = a.shape[0], a.shape[1]

    @classmethod
    def checker(cls, width, height, cells_x, cells_y, color0, color1):
        """Build a checkerboard texture."""
        data = np.zeros((height, width, 3), dtype=np.float64)
        for j in range(height):
            for i in range(width):
                cx = i * cells_x // width
                cy = j * cells_y // height
                data[j, i] = color0 if (cx + cy) % 2 == 0 else color1
        return cls(data)

    @staticmethod
    def _wrap_index(i, n, wrap):
        if wrap == "repeat":
            return i % n
        if wrap == "clamp":
            return min(max(i, 0), n - 1)
        raise ValueError("wrap must be 'clamp' or 'repeat'")

    def sample(self, u, v, mode="bilinear", wrap="clamp"):
        """Sample the texture at ``(u, v)``.

        ``mode`` is ``'nearest'`` or ``'bilinear'``; ``wrap`` is
        ``'clamp'`` or ``'repeat'`` and controls out-of-range behaviour
        on both axes (texture borders included).
        """
        if mode == "nearest":
            i = self._wrap_index(int(np.floor(u * self.width)), self.width, wrap)
            j = self._wrap_index(int(np.floor(v * self.height)), self.height, wrap)
            return self.data[j, i].copy()
        if mode != "bilinear":
            raise ValueError("mode must be 'nearest' or 'bilinear'")
        # Texel centers live at (i + 0.5) / size.
        x = u * self.width - 0.5
        y = v * self.height - 0.5
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        fx = x - x0
        fy = y - y0
        color = np.zeros(3, dtype=np.float64)
        for dy in (0, 1):
            for dx in (0, 1):
                ii = self._wrap_index(x0 + dx, self.width, wrap)
                jj = self._wrap_index(y0 + dy, self.height, wrap)
                w = (fx if dx else 1.0 - fx) * (fy if dy else 1.0 - fy)
                color += w * self.data[jj, ii]
        return color
