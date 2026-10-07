"""Pure-CPU software rasterizer.

No OpenGL / Vulkan / DirectX / GPU / external render services.
Everything lives in local memory (numpy arrays) or local files (PPM/PNG).
"""

from .math3d import (
    identity,
    translate,
    scale,
    rotate_x,
    rotate_y,
    rotate_z,
    look_at,
    perspective,
    transform_point,
)
from .clip import clip_triangle_near
from .texture import Texture
from .rasterizer import Framebuffer, Renderer, rasterize_triangle
from .imageio import write_ppm, write_png

__all__ = [
    "identity",
    "translate",
    "scale",
    "rotate_x",
    "rotate_y",
    "rotate_z",
    "look_at",
    "perspective",
    "transform_point",
    "clip_triangle_near",
    "Texture",
    "Framebuffer",
    "Renderer",
    "rasterize_triangle",
    "write_ppm",
    "write_png",
]
