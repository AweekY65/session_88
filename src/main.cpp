#include <cstdio>
#include <vector>

#include "rasterizer.h"

// Builds a cube centered at the origin with per-face UVs. Returns 36 vertices
// (12 triangles) in model space.
static std::vector<Vertex> makeCube() {
    const float p[8][3] = {
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
        {-1, -1,  1}, {1, -1,  1}, {1, 1,  1}, {-1, 1,  1},
    };
    const int faces[6][4] = {
        {4, 5, 6, 7}, // +Z
        {1, 0, 3, 2}, // -Z
        {5, 1, 2, 6}, // +X
        {0, 4, 7, 3}, // -X
        {7, 6, 2, 3}, // +Y
        {0, 1, 5, 4}, // -Y
    };
    const float uvs[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    std::vector<Vertex> out;
    for (const auto& f : faces) {
        for (int tri = 0; tri < 2; ++tri) {
            int idx[3] = {0, 1, 2};
            if (tri == 1) { idx[0] = 0; idx[1] = 2; idx[2] = 3; }
            for (int k = 0; k < 3; ++k) {
                Vertex v;
                v.position = {p[f[idx[k]]][0], p[f[idx[k]]][1], p[f[idx[k]]][2]};
                v.uv = {uvs[idx[k]][0], uvs[idx[k]][1]};
                v.color = {1, 1, 1};
                out.push_back(v);
            }
        }
    }
    return out;
}

static Texture makeChecker(int size, int cells) {
    Texture tex(size, size);
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            bool on = ((x * cells / size) + (y * cells / size)) % 2 == 0;
            tex.set(x, y, on ? Color{230, 220, 200} : Color{40, 60, 120});
        }
    return tex;
}

int main() {
    const int W = 512, H = 512;
    Framebuffer fb(W, H);
    fb.clear({32, 32, 48});
    Texture checker = makeChecker(256, 8);
    std::vector<Vertex> cube = makeCube();
    Rasterizer rast(fb);
    rast.texture = &checker;
    rast.sampleMode = SampleMode::Bilinear;
    rast.wrapMode = WrapMode::Repeat;
    rast.model = rotateY(0.7f) * rotateX(0.5f);
    rast.view = lookAt({0, 0, 4}, {0, 0, 0}, {0, 1, 0});
    rast.projection = perspective(1.05f, static_cast<float>(W) / H, 0.1f, 100.0f);
    rast.draw(cube.data(), cube.size());
    if (!fb.writePPM("output.ppm")) {
        std::fprintf(stderr, "failed to write output.ppm\n");
        return 1;
    }
    std::printf("wrote output.ppm (%dx%d), hash=%016llx\n", W, H,
                (unsigned long long)fb.hash());
    return 0;
}
