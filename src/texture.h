#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

struct Color {
    uint8_t r = 0, g = 0, b = 0;
};

enum class SampleMode { Nearest, Bilinear };
enum class WrapMode { Clamp, Repeat };

// RGB8 texture stored fully in memory.
class Texture {
public:
    Texture() = default;
    Texture(int w, int h) : width_(w), height_(h), data_(w * h * 3, 0) {}

    int width() const { return width_; }
    int height() const { return height_; }

    void set(int x, int y, Color c) {
        size_t i = (static_cast<size_t>(y) * width_ + x) * 3;
        data_[i] = c.r;
        data_[i + 1] = c.g;
        data_[i + 2] = c.b;
    }

    Color texel(int x, int y) const {
        size_t i = (static_cast<size_t>(y) * width_ + x) * 3;
        return {data_[i], data_[i + 1], data_[i + 2]};
    }

    Color sample(float u, float v, SampleMode mode, WrapMode wrap) const {
        u = wrapCoord(u, wrap);
        v = wrapCoord(v, wrap);
        if (mode == SampleMode::Nearest) {
            int x = std::min(static_cast<int>(u * width_), width_ - 1);
            int y = std::min(static_cast<int>(v * height_), height_ - 1);
            return texel(x, y);
        }
        // Bilinear: sample at texel centers.
        float fx = u * width_ - 0.5f;
        float fy = v * height_ - 0.5f;
        int x0 = static_cast<int>(std::floor(fx));
        int y0 = static_cast<int>(std::floor(fy));
        float tx = fx - x0;
        float ty = fy - y0;
        int x1 = x0 + 1, y1 = y0 + 1;
        if (wrap == WrapMode::Repeat) {
            x0 = mod(x0, width_);  x1 = mod(x1, width_);
            y0 = mod(y0, height_); y1 = mod(y1, height_);
        } else {
            x0 = std::clamp(x0, 0, width_ - 1);  x1 = std::clamp(x1, 0, width_ - 1);
            y0 = std::clamp(y0, 0, height_ - 1); y1 = std::clamp(y1, 0, height_ - 1);
        }
        Color c00 = texel(x0, y0), c10 = texel(x1, y0);
        Color c01 = texel(x0, y1), c11 = texel(x1, y1);
        auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
        Color out;
        out.r = toByte(lerp(lerp(c00.r, c10.r, tx), lerp(c01.r, c11.r, tx), ty));
        out.g = toByte(lerp(lerp(c00.g, c10.g, tx), lerp(c01.g, c11.g, tx), ty));
        out.b = toByte(lerp(lerp(c00.b, c10.b, tx), lerp(c01.b, c11.b, tx), ty));
        return out;
    }

private:
    static float wrapCoord(float t, WrapMode wrap) {
        if (wrap == WrapMode::Repeat) t -= std::floor(t);
        return std::clamp(t, 0.0f, 1.0f);
    }
    static int mod(int a, int n) { return ((a % n) + n) % n; }
    static uint8_t toByte(float v) {
        return static_cast<uint8_t>(std::clamp(std::lround(v), 0l, 255l));
    }

    int width_ = 0, height_ = 0;
    std::vector<uint8_t> data_; // RGB interleaved
};
