// A rectangular cloth built from a synthetic particle grid.
#pragma once
#include <cstdint>
#include <vector>

#include "pbd/math.h"

namespace pbd
{

enum class ConstraintKind : std::uint8_t
{
    Structural,
    Shear,
    Bend
};

struct DistanceConstraint
{
    int i, j;        // particle indices
    float rest;      // rest length
    float stiffness; // 0..1, scales the projection
};

enum class PinMode
{
    TopEdge,
    TopCorners,
    Frame
}; // Frame: all four edges, like a drum skin

struct ClothDesc
{
    int nx = 96;           // particles along the width
    int ny = 96;           // particles along the height
    float spacing = 0.02f; // rest distance between neighbours (m)
    bool shear = true;     // add diagonal constraints
    bool bend = true;      // add skip-one constraints
    float stretchStiffness = 1.0f;
    float shearStiffness = 0.8f;
    float bendStiffness = 0.3f;
    PinMode pin = PinMode::TopEdge;
};

struct Cloth
{
    int nx = 0, ny = 0;
    std::vector<Vec3> position; // positions
    std::vector<Vec3> velocity; // velocities
    std::vector<float> invMass; // inverse masses (0 = pinned)
    std::vector<DistanceConstraint> constraints;
    std::vector<ConstraintKind> kinds;

    int index(int i, int j) const
    {
        return j * nx + i;
    }
    int particleCount() const
    {
        return static_cast<int>(position.size());
    }
    int constraintCount() const
    {
        return static_cast<int>(constraints.size());
    }

    // Builds a vertical cloth hanging in the XY plane; row 0 is the top edge.
    static Cloth makeGrid(const ClothDesc& desc);

    // Largest relative constraint violation over the whole cloth.
    float maxRelativeError() const;
};

} // namespace pbd
