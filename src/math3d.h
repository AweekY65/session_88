#pragma once

#include <cmath>

namespace sr {

constexpr float kPi = 3.14159265358979323846f;

inline float radians(float degrees) { return degrees * kPi / 180.0f; }

struct Vec2 { float x = 0.0f, y = 0.0f; };
struct Vec3 { float x = 0.0f, y = 0.0f, z = 0.0f; };
struct Vec4 { float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f; };

inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 normalize(const Vec3& v) {
    float len = std::sqrt(dot(v, v));
    return {v.x / len, v.y / len, v.z / len};
}

struct Mat4 {
    float m[4][4] = {};
    static Mat4 identity() {
        Mat4 r;
        for (int i = 0; i < 4; ++i) r.m[i][i] = 1.0f;
        return r;
    }
    const float* operator[](int row) const { return m[row]; }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a.m[i][k] * b.m[k][j];
            r.m[i][j] = s;
        }
    return r;
}

inline Vec4 operator*(const Mat4& m, const Vec4& v) {
    return {
        m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3] * v.w,
        m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3] * v.w,
        m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3] * v.w,
        m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3] * v.w,
    };
}

inline Mat4 translation(float tx, float ty, float tz) {
    Mat4 r = Mat4::identity();
    r.m[0][3] = tx;
    r.m[1][3] = ty;
    r.m[2][3] = tz;
    return r;
}

inline Mat4 scaling(float sx, float sy, float sz) {
    Mat4 r;
    r.m[0][0] = sx;
    r.m[1][1] = sy;
    r.m[2][2] = sz;
    r.m[3][3] = 1.0f;
    return r;
}

inline Mat4 rotationX(float rad) {
    Mat4 r = Mat4::identity();
    float c = std::cos(rad), s = std::sin(rad);
    r.m[1][1] = c;  r.m[1][2] = -s;
    r.m[2][1] = s;  r.m[2][2] = c;
    return r;
}

inline Mat4 rotationY(float rad) {
    Mat4 r = Mat4::identity();
    float c = std::cos(rad), s = std::sin(rad);
    r.m[0][0] = c;  r.m[0][2] = s;
    r.m[2][0] = -s; r.m[2][2] = c;
    return r;
}

inline Mat4 rotationZ(float rad) {
    Mat4 r = Mat4::identity();
    float c = std::cos(rad), s = std::sin(rad);
    r.m[0][0] = c;  r.m[0][1] = -s;
    r.m[1][0] = s;  r.m[1][1] = c;
    return r;
}

// Right-handed perspective projection. Camera looks down -Z, NDC z in [-1, 1].
inline Mat4 perspective(float fovyRad, float aspect, float zNear, float zFar) {
    Mat4 r;
    float f = 1.0f / std::tan(fovyRad * 0.5f);
    r.m[0][0] = f / aspect;
    r.m[1][1] = f;
    r.m[2][2] = (zFar + zNear) / (zNear - zFar);
    r.m[2][3] = 2.0f * zFar * zNear / (zNear - zFar);
    r.m[3][2] = -1.0f;
    return r;
}

// Right-handed lookAt view matrix.
inline Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);
    Mat4 r = Mat4::identity();
    r.m[0][0] = s.x;  r.m[0][1] = s.y;  r.m[0][2] = s.z;  r.m[0][3] = -dot(s, eye);
    r.m[1][0] = u.x;  r.m[1][1] = u.y;  r.m[1][2] = u.z;  r.m[1][3] = -dot(u, eye);
    r.m[2][0] = -f.x; r.m[2][1] = -f.y; r.m[2][2] = -f.z; r.m[2][3] = dot(f, eye);
    return r;
}

} // namespace sr
