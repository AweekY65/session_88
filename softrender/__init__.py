"""Pure-CPU software rasterization renderer (no GPU, no external services)."""
from . import math3d
from .imageio import encode_png, save_png, save_ppm, to_uint8
from .rasterizer import (
    Framebuffer,
    Rasterizer,
    Vertex,
    barycentric,
    clip_triangle_near,
    render,
)
from .texture import Texture

__all__ = [
    "math3d",
    "Texture",
    "Framebuffer",
    "Rasterizer",
    "Vertex",
    "barycentric",
    "clip_triangle_near",
    "render",
    "save_ppm",
    "save_png",
    "encode_png",
    "to_uint8",
]
