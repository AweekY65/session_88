#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "framebuffer.h"
#include "math3d.h"
#include "texture.h"

struct Vertex {
    Vec3 position;       // model space
    Vec2 uv;             // texture coordinates
    Vec3 color{1, 1, 1}; // vertex color in [0,1]
};

// Vertex after the vertex shader (clip space, before perspective divide).
struct ClipVertex {
    Vec4 clip;
    Vec2 uv;
    Vec3 color;
};

inline ClipVertex lerp(const ClipVertex& a, const ClipVertex& b, float t) {
    auto l = [t](float x, float y) { return x + (y - x) * t; };
    ClipVertex r;
    r.clip = {l(a.clip.x, b.clip.x), l(a.clip.y, b.clip.y),
              l(a.clip.z, b.clip.z), l(a.clip.w, b.clip.w)};
    r.uv = {l(a.uv.x, b.uv.x), l(a.uv.y, b.uv.y)};
    r.color = {l(a.color.x, b.color.x), l(a.color.y, b.color.y), l(a.color.z, b.color.z)};
    return r;
}

// Sutherland-Hodgman clip of a triangle against the near plane z >= -w
// (clip space). Returns a convex polygon of 0..4 vertices in `out`.
inline int clipNearPlane(const ClipVertex tri[3], ClipVertex out[4]) {
    auto dist = [](const ClipVertex& v) { return v.clip.z + v.clip.w; };
    int n = 0;
    for (int i = 0; i < 3; ++i) {
        const ClipVertex& a = tri[i];
        const ClipVertex& b = tri[(i + 1) % 3];
        float da = dist(a), db = dist(b);
        bool inA = da >= 0.0f, inB = db >= 0.0f;
        if (inA) out[n++] = a;
        if (inA != inB) {
            float t = da / (da - db);
            out[n++] = lerp(a, b, t);
        }
    }
    return n;
}

class Rasterizer {
public:
    explicit Rasterizer(Framebuffer& fb) : fb_(fb) {}

    Mat4 model = Mat4::identity();
    Mat4 view = Mat4::identity();
    Mat4 projection = Mat4::identity();

    const Texture* texture = nullptr;
    SampleMode sampleMode = SampleMode::Bilinear;
    WrapMode wrapMode = WrapMode::Clamp;

    // Draws `count` vertices as a triangle list through the full pipeline:
    // model/view/projection -> near-plane clip -> divide -> rasterize.
    void draw(const Vertex* verts, size_t count) {
        Mat4 mvp = projection * view * model;
        for (size_t i = 0; i + 3 <= count; i += 3) {
            ClipVertex tri[3];
            for (int k = 0; k < 3; ++k) {
                const Vertex& v = verts[i + k];
                tri[k].clip = mvp * Vec4{v.position.x, v.position.y, v.position.z, 1.0f};
                tri[k].uv = v.uv;
                tri[k].color = v.color;
            }
            ClipVertex poly[4];
            int n = clipNearPlane(tri, poly);
            for (int t = 0; t + 2 < n; ++t)
                rasterizeTriangle(poly[0], poly[t + 1], poly[t + 2]);
        }
    }

private:
    struct ScreenVertex {
        float x, y;   // screen space (pixels, y down)
        float z;      // NDC depth in [-1,1]
        float invW;   // 1 / clip.w
        Vec2 uv;
        Vec3 color;
    };

    static float edge(const ScreenVertex& a, const ScreenVertex& b, float px, float py) {
        return (px - a.x) * (b.y - a.y) - (py - a.y) * (b.x - a.x);
    }

    // Top-left fill rule: an edge is top-left if it is a horizontal edge going
    // left, or a non-horizontal edge going down (screen space, y down).
    static bool isTopLeft(const ScreenVertex& a, const ScreenVertex& b) {
        return (a.y == b.y && a.x > b.x) || (a.y < b.y);
    }

    void rasterizeTriangle(const ClipVertex& cv0, const ClipVertex& cv1,
                           const ClipVertex& cv2) {
        const ClipVertex* cvs[3] = {&cv0, &cv1, &cv2};
        ScreenVertex s[3];
        for (int i = 0; i < 3; ++i) {
            const Vec4& c = cvs[i]->clip;
            float invW = 1.0f / c.w;
            float ndcX = c.x * invW, ndcY = c.y * invW, ndcZ = c.z * invW;
            s[i].x = (ndcX * 0.5f + 0.5f) * fb_.width();
            s[i].y = (0.5f - ndcY * 0.5f) * fb_.height();
            s[i].z = ndcZ;
            s[i].invW = invW;
            s[i].uv = cvs[i]->uv;
            s[i].color = cvs[i]->color;
        }

        float area = edge(s[0], s[1], s[2].x, s[2].y);
        if (area == 0.0f) return;
        if (area < 0.0f) {
            std::swap(s[1], s[2]);
            area = -area;
        }

        int minX = std::max(0, static_cast<int>(std::floor(std::min({s[0].x, s[1].x, s[2].x}))));
        int maxX = std::min(fb_.width() - 1, static_cast<int>(std::ceil(std::max({s[0].x, s[1].x, s[2].x}))));
        int minY = std::max(0, static_cast<int>(std::floor(std::min({s[0].y, s[1].y, s[2].y}))));
        int maxY = std::min(fb_.height() - 1, static_cast<int>(std::ceil(std::max({s[0].y, s[1].y, s[2].y}))));

        bool tl0 = isTopLeft(s[1], s[2]);
        bool tl1 = isTopLeft(s[2], s[0]);
        bool tl2 = isTopLeft(s[0], s[1]);

        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                float px = x + 0.5f, py = y + 0.5f;
                float e0 = edge(s[1], s[2], px, py);
                float e1 = edge(s[2], s[0], px, py);
                float e2 = edge(s[0], s[1], px, py);
                if (e0 < 0.0f || (e0 == 0.0f && !tl0)) continue;
                if (e1 < 0.0f || (e1 == 0.0f && !tl1)) continue;
                if (e2 < 0.0f || (e2 == 0.0f && !tl2)) continue;

                // Barycentric coordinates.
                float w0 = e0 / area, w1 = e1 / area, w2 = e2 / area;

                // Depth: NDC z interpolates linearly in screen space.
                float z = w0 * s[0].z + w1 * s[1].z + w2 * s[2].z;
                if (z < -1.0f || z > 1.0f) continue;

                // Perspective-correct interpolation: attributes are linear in
                // eye space, so interpolate attr/w and 1/w, then divide.
                float invW = w0 * s[0].invW + w1 * s[1].invW + w2 * s[2].invW;
                float u = (w0 * s[0].uv.x * s[0].invW + w1 * s[1].uv.x * s[1].invW +
                           w2 * s[2].uv.x * s[2].invW) / invW;
                float v = (w0 * s[0].uv.y * s[0].invW + w1 * s[1].uv.y * s[1].invW +
                           w2 * s[2].uv.y * s[2].invW) / invW;
                Vec3 col;
                col.x = (w0 * s[0].color.x * s[0].invW + w1 * s[1].color.x * s[1].invW +
                         w2 * s[2].color.x * s[2].invW) / invW;
                col.y = (w0 * s[0].color.y * s[0].invW + w1 * s[1].color.y * s[1].invW +
                         w2 * s[2].color.y * s[2].invW) / invW;
                col.z = (w0 * s[0].color.z * s[0].invW + w1 * s[1].color.z * s[1].invW +
                         w2 * s[2].color.z * s[2].invW) / invW;

                float r = col.x, g = col.y, b = col.z;
                if (texture) {
                    Color tex = texture->sample(u, v, sampleMode, wrapMode);
                    r *= tex.r / 255.0f;
                    g *= tex.g / 255.0f;
                    b *= tex.b / 255.0f;
                }
                Color out{
                    static_cast<uint8_t>(std::clamp(std::lround(r * 255.0f), 0l, 255l)),
                    static_cast<uint8_t>(std::clamp(std::lround(g * 255.0f), 0l, 255l)),
                    static_cast<uint8_t>(std::clamp(std::lround(b * 255.0f), 0l, 255l)),
                };
                fb_.writePixel(x, y, z, out);
            }
        }
    }

    Framebuffer& fb_;
};
