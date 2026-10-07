"""Render a small demo scene to demo.ppm / demo.png (pure CPU, no window).

Run from the repository root:

    python3 examples/render_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from softrenderer import Renderer, Texture, look_at, rotate_y, translate


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    r = Renderer(256, 256, fovy=np.radians(60.0), near=0.1, far=100.0,
                 clear_color=(0.1, 0.1, 0.15))
    view = look_at(eye=(0, 1.2, 3.5), center=(0, -0.2, 0), up=(0, 1, 0))

    # checkerboard floor (perspective-correct interpolation showcase)
    floor_pos = np.array([
        [-5, -1, 3], [5, -1, 3], [5, -1, -8], [-5, -1, -8],
    ], dtype=float)
    floor_uv = np.array([[0, 0], [6, 0], [6, 6], [0, 6]], dtype=float)
    floor_col = np.ones((4, 3))
    checker = Texture.checker(64, 64, 8, 8, (0.95, 0.95, 0.95), (0.15, 0.15, 0.2))
    r.render(floor_pos, floor_uv, floor_col, np.array([[0, 1, 2], [0, 2, 3]]),
             view=view, texture=checker, sample_mode="bilinear", wrap="repeat")

    # a rotating triangle fan (three overlapping triangles, z-buffer demo)
    colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.4, 1.0)]
    for k, color in enumerate(colors):
        angle = np.radians(30 * k)
        model = translate(-0.8 + 0.8 * k, 0.0, -0.5 * k) @ rotate_y(angle)
        pos = np.array([[-0.7, -0.7, 0.0], [0.7, -0.7, 0.0], [0.0, 0.9, 0.0]])
        r.render(pos, np.zeros((3, 2)), np.array([color] * 3),
                 np.array([[0, 1, 2]]), model=model, view=view)

    ppm_path = os.path.join(out_dir, "demo.ppm")
    png_path = os.path.join(out_dir, "demo.png")
    r.fb.save_ppm(ppm_path)
    r.fb.save_png(png_path)
    print("wrote", ppm_path, "and", png_path)


if __name__ == "__main__":
    main()
