// Orbit camera, column-major 4x4 matrices and screen-ray helpers.
#pragma once
#include <cmath>

#include "pbd/math.h"

namespace viewer
{
using pbd::Vec3;

inline Vec3 Cross(Vec3 a, Vec3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline Vec3 Normalize(Vec3 v)
{
    const float len = pbd::length(v);
    return len > 1e-12f ? v * (1.0f / len) : Vec3{0, 0, 0};
}

struct Mat4
{
    float m[16]; // column-major, as OpenGL expects

    static Mat4 Identity()
    {
        Mat4 r = {};
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    static Mat4 Perspective(float fovyRadians, float aspect, float zNear, float zFar)
    {
        Mat4 r = {};
        const float f = 1.0f / std::tan(fovyRadians * 0.5f);
        r.m[0] = f / aspect;
        r.m[5] = f;
        r.m[10] = (zFar + zNear) / (zNear - zFar);
        r.m[11] = -1.0f;
        r.m[14] = 2.0f * zFar * zNear / (zNear - zFar);
        return r;
    }

    static Mat4 LookAt(Vec3 eye, Vec3 target, Vec3 up)
    {
        const Vec3 f = Normalize(target - eye);
        const Vec3 s = Normalize(Cross(f, up));
        const Vec3 u = Cross(s, f);
        Mat4 r = Identity();
        r.m[0] = s.x;
        r.m[4] = s.y;
        r.m[8] = s.z;
        r.m[1] = u.x;
        r.m[5] = u.y;
        r.m[9] = u.z;
        r.m[2] = -f.x;
        r.m[6] = -f.y;
        r.m[10] = -f.z;
        r.m[12] = -pbd::dot(s, eye);
        r.m[13] = -pbd::dot(u, eye);
        r.m[14] = pbd::dot(f, eye);
        return r;
    }

    Mat4 operator*(const Mat4& b) const
    {
        Mat4 r = {};
        for (int c = 0; c < 4; ++c)
        {
            for (int row = 0; row < 4; ++row)
            {
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k)
                {
                    sum += m[k * 4 + row] * b.m[c * 4 + k];
                }
                r.m[c * 4 + row] = sum;
            }
        }
        return r;
    }
};

struct Ray
{
    Vec3 origin;
    Vec3 dir; // unit length
};

class OrbitCamera
{
  public:
    Vec3 target{0.0f, 0.0f, 0.0f};
    float yaw = 0.55f;    // radians around +Y
    float pitch = 0.32f;  // radians above the horizon
    float distance = 3.0f;
    float fovy = 0.80f;   // radians

    Vec3 Eye() const
    {
        const float cp = std::cos(pitch);
        return target + Vec3{distance * cp * std::sin(yaw), distance * std::sin(pitch), distance * cp * std::cos(yaw)};
    }
    Vec3 Forward() const
    {
        return Normalize(target - Eye());
    }
    Vec3 Right() const
    {
        return Normalize(Cross(Forward(), Vec3{0, 1, 0}));
    }
    Vec3 Up() const
    {
        return Cross(Right(), Forward());
    }

    Mat4 View() const
    {
        return Mat4::LookAt(Eye(), target, Vec3{0, 1, 0});
    }
    Mat4 Projection(float aspect) const
    {
        return Mat4::Perspective(fovy, aspect, 0.02f, 100.0f);
    }

    void Orbit(float dx, float dy)
    {
        yaw -= dx * 0.008f;
        pitch += dy * 0.008f;
        const float limit = 1.45f;
        pitch = pitch < -limit ? -limit : (pitch > limit ? limit : pitch);
    }
    void Zoom(float wheelSteps)
    {
        distance *= std::pow(0.88f, wheelSteps);
        distance = distance < 0.3f ? 0.3f : (distance > 30.0f ? 30.0f : distance);
    }

    // Ray through pixel (px, py) of a width x height viewport (origin top-left).
    Ray ScreenRay(float px, float py, int width, int height) const
    {
        const float aspect = static_cast<float>(width) / static_cast<float>(height > 0 ? height : 1);
        const float t = std::tan(fovy * 0.5f);
        const float nx = (2.0f * px / static_cast<float>(width) - 1.0f) * t * aspect;
        const float ny = (1.0f - 2.0f * py / static_cast<float>(height)) * t;
        return Ray{Eye(), Normalize(Forward() + Right() * nx + Up() * ny)};
    }
};

// Squared distance from point p to the ray (points behind the origin use the origin).
inline float DistanceToRaySq(const Ray& r, Vec3 p)
{
    const Vec3 d = p - r.origin;
    float t = pbd::dot(d, r.dir);
    t = t < 0.0f ? 0.0f : t;
    const Vec3 q = r.origin + r.dir * t - p;
    return pbd::dot(q, q);
}

// Intersection of a ray with the plane through `point` with normal `n`.
inline bool IntersectPlane(const Ray& r, Vec3 point, Vec3 n, Vec3& hit)
{
    const float denom = pbd::dot(r.dir, n);
    if (std::fabs(denom) < 1e-6f)
    {
        return false;
    }
    const float t = pbd::dot(point - r.origin, n) / denom;
    if (t < 0.0f)
    {
        return false;
    }
    hit = r.origin + r.dir * t;
    return true;
}
} // namespace viewer
