// Minimal vector math shared by host (C++) and device (CUDA) code.
#pragma once
#include <cmath>

#if defined(__CUDACC__)
#define PBD_HD __host__ __device__ __forceinline__
#else
#define PBD_HD inline
#endif

namespace pbd
{

struct Vec3
{
    float x, y, z;
};

PBD_HD Vec3 make_vec3(float x, float y, float z)
{
    return Vec3{x, y, z};
}
PBD_HD Vec3 operator+(Vec3 a, Vec3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
PBD_HD Vec3 operator-(Vec3 a, Vec3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
PBD_HD Vec3 operator*(Vec3 a, float s)
{
    return {a.x * s, a.y * s, a.z * s};
}
PBD_HD Vec3 operator*(float s, Vec3 a)
{
    return a * s;
}
PBD_HD Vec3& operator+=(Vec3& a, Vec3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
PBD_HD Vec3& operator-=(Vec3& a, Vec3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}
PBD_HD float dot(Vec3 a, Vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
PBD_HD float length(Vec3 a)
{
    return sqrtf(dot(a, a));
}

// Position-based distance constraint C(p_i, p_j) = |p_j - p_i| - rest.
// Returns the corrections for both endpoints (already weighted by inverse mass).
// The same function is used by the CPU and GPU solvers, so both projections are identical.
PBD_HD bool project_distance(Vec3 pi, Vec3 pj, float wi, float wj, float rest, float stiffness, Vec3& dpi,
                             Vec3& dpj)
{
    const float wsum = wi + wj;
    if (wsum <= 0.0f)
    {
        return false;
    }
    const Vec3 d = pj - pi;
    const float len = length(d);
    if (len < 1e-9f)
    {
        return false;
    }
    const Vec3 n = d * (1.0f / len);
    const float c = len - rest;
    const float s = stiffness * c / wsum;
    dpi = n * (wi * s);  // p_i moves toward p_j when stretched
    dpj = n * (-wj * s); // p_j moves toward p_i
    return true;
}

// Relative violation |C| / rest, the error metric used for convergence checks.
PBD_HD float relative_error(Vec3 pi, Vec3 pj, float rest)
{
    return fabsf(length(pj - pi) - rest) / rest;
}

} // namespace pbd
