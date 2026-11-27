// Block partition of the particle grid and a block -> constraint lookup table.
//
// The grid is cut into B x B particle tiles. Every constraint is registered in the
// blocks of *both* endpoints, so a constraint that straddles two tiles is still
// found when only one of them is active. The table is stored in CSR form
// (start offsets + flat index array) so it can be uploaded to the GPU as-is.
#pragma once
#include <vector>

#include "pbd/cloth.h"

namespace pbd
{

struct BlockGrid
{
    int blockSize = 8;  // particles per tile edge
    int bx = 0, by = 0; // tiles along x and y

    static BlockGrid make(int nx, int ny, int blockSize);
    int count() const
    {
        return bx * by;
    }
    int blockOf(int i, int j) const
    {
        return (j / blockSize) * bx + (i / blockSize);
    }
};

struct BlockConstraintTable
{
    std::vector<int> start;       // size count()+1, CSR offsets
    std::vector<int> constraints; // flat constraint indices

    static BlockConstraintTable build(const Cloth& cloth, const BlockGrid& grid);
    int sizeOf(int block) const
    {
        return start[block + 1] - start[block];
    }
};

// Marks the 8-neighbourhood of every active block (one-tile halo) as active too.
std::vector<unsigned char> dilateBlocks(const BlockGrid& grid, const std::vector<unsigned char>& active);

// Host reference for the GPU gather: unique constraints touching the active blocks.
std::vector<int> gatherConstraints(const BlockConstraintTable& table, const std::vector<unsigned char>& active);

} // namespace pbd
