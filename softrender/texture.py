"""Texture storage and sampling (nearest / bilinear), pure CPU."""
import numpy as np


class Texture:
    """RGB texture stored as a float64 HxWx3 numpy array in [0, 1].

    wrap: 'clamp' (clamp to edge) or 'repeat' (tile).
    """

    def __init__(self, data, wrap="clamp"):
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("texture must be HxW or HxWx3")
        if arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValueError("texture must be at least 1x1")
        self.data = np.clip(arr, 0.0, 1.0)
        if wrap not in ("clamp", "repeat"):
            raise ValueError("wrap must be 'clamp' or 'repeat'")
        self.wrap = wrap

    @property
    def width(self):
        return self.data.shape[1]

    @property
    def height(self):
        return self.data.shape[0]

    def _wrap_coord(self, t, n):
        if self.wrap == "repeat":
            return int(t % n)
        return int(min(max(t, 0.0), n - 1.0))

    def sample_nearest(self, u, v):
        """Nearest-neighbor sampling; uv in texture space, (0,0) = texel (0,0) center."""
        x = self._wrap_coord(int(np.floor(u * self.width)), self.width)
        y = self._wrap_coord(int(np.floor(v * self.height)), self.height)
        return self.data[y, x].copy()

    def sample_bilinear(self, u, v):
        """Bilinear sampling at continuous texel coordinates.

        uv is mapped so that texel (i, j) center sits at
        ((i + 0.5) / W, (j + 0.5) / H). Boundary texels are clamped/wrapped
        according to the wrap mode.
        """
        fx = u * self.width - 0.5
        fy = v * self.height - 0.5
        x0 = int(np.floor(fx))
        y0 = int(np.floor(fy))
        tx = fx - x0
        ty = fy - y0
        c = {}
        for dx in (0, 1):
            for dy in (0, 1):
                xi = self._wrap_coord(x0 + dx, self.width)
                yi = self._wrap_coord(y0 + dy, self.height)
                c[(dx, dy)] = self.data[yi, xi]
        top = c[(0, 0)] * (1.0 - tx) + c[(1, 0)] * tx
        bot = c[(0, 1)] * (1.0 - tx) + c[(1, 1)] * tx
        return top * (1.0 - ty) + bot * ty

    def sample(self, u, v, mode="bilinear"):
        if mode == "nearest":
            return self.sample_nearest(u, v)
        if mode == "bilinear":
            return self.sample_bilinear(u, v)
        raise ValueError("mode must be 'nearest' or 'bilinear'")
