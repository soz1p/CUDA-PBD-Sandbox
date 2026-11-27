#include "pbd/cloth.h"

#include <algorithm>

namespace pbd
{

static void AddConstraint(Cloth& cloth, int first, int second, float stiffness, ConstraintKind kind)
{
    DistanceConstraint constraint;
    constraint.i = first;
    constraint.j = second;
    constraint.rest = length(cloth.position[second] - cloth.position[first]);
    constraint.stiffness = stiffness;
    cloth.constraints.push_back(constraint);
    cloth.kinds.push_back(kind);
}

Cloth Cloth::makeGrid(const ClothDesc& desc)
{
    Cloth cloth;
    cloth.nx = desc.nx;
    cloth.ny = desc.ny;
    const int n = desc.nx * desc.ny;
    cloth.position.resize(n);
    cloth.velocity.assign(n, Vec3{0, 0, 0});
    cloth.invMass.assign(n, 1.0f);

    const float halfWidth = 0.5f * desc.spacing * static_cast<float>(desc.nx - 1);
    for (int j = 0; j < desc.ny; ++j)
    {
        for (int i = 0; i < desc.nx; ++i)
        {
            cloth.position[cloth.index(i, j)] = Vec3{i * desc.spacing - halfWidth, -j * desc.spacing, 0.0f};
        }
    }

    if (desc.pin == PinMode::TopEdge)
    {
        for (int i = 0; i < desc.nx; ++i)
        {
            cloth.invMass[cloth.index(i, 0)] = 0.0f;
        }
    }
    else if (desc.pin == PinMode::Frame)
    {
        for (int i = 0; i < desc.nx; ++i)
        {
            cloth.invMass[cloth.index(i, 0)] = 0.0f;
            cloth.invMass[cloth.index(i, desc.ny - 1)] = 0.0f;
        }
        for (int j = 0; j < desc.ny; ++j)
        {
            cloth.invMass[cloth.index(0, j)] = 0.0f;
            cloth.invMass[cloth.index(desc.nx - 1, j)] = 0.0f;
        }
    }
    else
    {
        cloth.invMass[cloth.index(0, 0)] = 0.0f;
        cloth.invMass[cloth.index(desc.nx - 1, 0)] = 0.0f;
    }

    for (int j = 0; j < desc.ny; ++j)
    {
        for (int i = 0; i < desc.nx; ++i)
        {
            const int p = cloth.index(i, j);
            if (i + 1 < desc.nx)
            {
                AddConstraint(cloth, p, cloth.index(i + 1, j), desc.stretchStiffness, ConstraintKind::Structural);
            }
            if (j + 1 < desc.ny)
            {
                AddConstraint(cloth, p, cloth.index(i, j + 1), desc.stretchStiffness, ConstraintKind::Structural);
            }
            if (desc.shear && i + 1 < desc.nx && j + 1 < desc.ny)
            {
                AddConstraint(cloth, p, cloth.index(i + 1, j + 1), desc.shearStiffness, ConstraintKind::Shear);
                AddConstraint(cloth, cloth.index(i + 1, j), cloth.index(i, j + 1), desc.shearStiffness,
                              ConstraintKind::Shear);
            }
            if (desc.bend && i + 2 < desc.nx)
            {
                AddConstraint(cloth, p, cloth.index(i + 2, j), desc.bendStiffness, ConstraintKind::Bend);
            }
            if (desc.bend && j + 2 < desc.ny)
            {
                AddConstraint(cloth, p, cloth.index(i, j + 2), desc.bendStiffness, ConstraintKind::Bend);
            }
        }
    }
    return cloth;
}

float Cloth::maxRelativeError() const
{
    float e = 0.0f;
    for (int i = 0; i < constraintCount(); i++)
    {
        const DistanceConstraint& edge = constraints[i];
        e = std::max(e, relative_error(position[edge.i], position[edge.j], edge.rest));
    }
    return e;
}

} // namespace pbd
