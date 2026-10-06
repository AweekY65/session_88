#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "texture.h"

// CPU-only color + depth framebuffer. Color is RGB8, depth is float NDC z.
class Framebuffer {
public:
    Framebuffer(int w, int h) : width_(w), height_(h) {
        color_.assign(static_cast<size_t>(w) * h * 3, 0);
        depth_.assign(static_cast<size_t>(w) * h, 1.0f);
    }

    int width() const { return width_; }
    int height() const { return height_; }

    void clear(Color c, float depth = 1.0f) {
        for (size_t i = 0; i + 2 < color_.size(); i += 3) {
            color_[i] = c.r;
            color_[i + 1] = c.g;
            color_[i + 2] = c.b;
        }
        std::fill(depth_.begin(), depth_.end(), depth);
    }

    float depthAt(int x, int y) const {
        return depth_[static_cast<size_t>(y) * width_ + x];
    }

    Color pixelAt(int x, int y) const {
        size_t i = (static_cast<size_t>(y) * width_ + x) * 3;
        return {color_[i], color_[i + 1], color_[i + 2]};
    }

    // Writes only if the fragment passes the depth test (less wins).
    bool writePixel(int x, int y, float depth, Color c) {
        size_t idx = static_cast<size_t>(y) * width_ + x;
        if (depth >= depth_[idx]) return false;
        depth_[idx] = depth;
        color_[idx * 3] = c.r;
        color_[idx * 3 + 1] = c.g;
        color_[idx * 3 + 2] = c.b;
        return true;
    }

    bool writePPM(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) return false;
        out << "P6\n" << width_ << " " << height_ << "\n255\n";
        out.write(reinterpret_cast<const char*>(color_.data()),
                  static_cast<std::streamsize>(color_.size()));
        return out.good();
    }

    // FNV-1a 64-bit hash over the color buffer, for golden-image tests.
    uint64_t hash() const {
        uint64_t h = 14695981039346656037ull;
        for (uint8_t byte : color_) {
            h ^= byte;
            h *= 1099511628211ull;
        }
        return h;
    }

private:
    int width_, height_;
    std::vector<uint8_t> color_;
    std::vector<float> depth_;
};
