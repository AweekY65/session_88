"""Render the demo scene to out/demo.ppm and out/demo.png (no window)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from scene import render_scene  # noqa: E402
from softrender import save_png, save_ppm  # noqa: E402


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "out")
    os.makedirs(out_dir, exist_ok=True)
    r = render_scene(256, 256)
    ppm_path = os.path.join(out_dir, "demo.ppm")
    png_path = os.path.join(out_dir, "demo.png")
    save_ppm(ppm_path, r.framebuffer.color)
    save_png(png_path, r.framebuffer.color)
    print(f"wrote {ppm_path} and {png_path}")


if __name__ == "__main__":
    main()
