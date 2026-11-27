#include "pbd/blocks.h"

#include <algorithm>

namespace pbd
{

BlockGrid BlockGrid::make(int nx, int ny, int blockSize)
{
    BlockGrid grid;
    grid.blockSize = blockSize;
    grid.bx = (nx + blockSize - 1) / blockSize;
    grid.by = (ny + blockSize - 1) / blockSize;
    return grid;
}

BlockConstraintTable BlockConstraintTable::build(const Cloth& cloth, const BlockGrid& grid)
{
    const int blockCount = grid.count();

    // Pass 1: count registrations per block.
    std::vector<int> counts(blockCount, 0);
    for (int k = 0; k < cloth.constraintCount(); k++)
    {
        const DistanceConstraint& c = cloth.constraints[k];
        const int a = grid.blockOf(c.i % cloth.nx, c.i / cloth.nx),
                  b = grid.blockOf(c.j % cloth.nx, c.j / cloth.nx);
        counts[a]++;
        if (b != a)
        {
            counts[b]++;
        }
    }

    BlockConstraintTable table;
    table.start.assign(blockCount + 1, 0);
    for (int b = 0; b < blockCount; ++b)
    {
        table.start[b + 1] = table.start[b] + counts[b];
    }
    table.constraints.resize(table.start[blockCount]);

    // Pass 2: scatter constraint ids into their block ranges.
    std::vector<int> cursor(table.start.begin(), table.start.end() - 1);
    for (int k = 0; k < cloth.constraintCount(); ++k)
    {
        const DistanceConstraint& c = cloth.constraints[k];
        const int a = grid.blockOf(c.i % cloth.nx, c.i / cloth.nx),
                  b = grid.blockOf(c.j % cloth.nx, c.j / cloth.nx);
        table.constraints[cursor[a]++] = k;
        if (b != a)
        {
            table.constraints[cursor[b]++] = k;
        }
    }
    return table;
}

std::vector<unsigned char> dilateBlocks(const BlockGrid& grid, const std::vector<unsigned char>& active)
{
    std::vector<unsigned char> out(active.size(), 0);
    for (int y = 0; y < grid.by; ++y)
    {
        for (int x = 0; x < grid.bx; ++x)
        {
            if (!active[y * grid.bx + x])
            {
                continue;
            }
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const int X = x + dx, Y = y + dy;
                    if (X >= 0 && Y >= 0 && X < grid.bx && Y < grid.by)
                    {
                        out[Y * grid.bx + X] = 1;
                    }
                }
            }
        }
    }
    return out;
}

std::vector<int> gatherConstraints(const BlockConstraintTable& table, const std::vector<unsigned char>& active)
{
    std::vector<int> out;
    std::vector<unsigned char> seen;
    int maxId = -1;
    for (size_t k = 0; k < table.constraints.size(); k++)
    {
        maxId = std::max(maxId, table.constraints[k]);
    }
    seen.assign(static_cast<size_t>(maxId + 1), 0);
    for (int b = 0; b + 1 < static_cast<int>(table.start.size()); ++b)
    {
        if (!active[b])
        {
            continue;
        }
        for (int k = table.start[b]; k < table.start[b + 1]; ++k)
        {
            const int c = table.constraints[k];
            if (!seen[c])
            {
                seen[c] = 1;
                out.push_back(c);
            }
        }
    }
    return out;
}

} // namespace pbd
