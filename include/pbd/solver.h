#pragma once
#include "pbd/blocks.h"

#include <vector>

#include "pbd/cloth.h"

namespace pbd
{

struct SolverConfig
{
    int substeps = 8;        // each frame is split into substeps (small steps converge far better)
    int maxIterations = 10;  // upper bound of Jacobi iterations per substep
    int checkEvery = 5;      // evaluate the error every N iterations
    float tolerance = 1e-2f; // stop a substep early when max |C|/rest falls below this
    float relaxation = 1.5f; // Jacobi over-relaxation factor (1..2)
    float damping = 0.01f;   // velocity damping per frame (split across substeps)
    Vec3 gravity{0.0f, -9.81f, 0.0f};
    // Active-block solver only
    int blockSize = 8;              // B x B particles per block
    // A block may sleep only when all of these hold for `sleepFrames` frames in a row:
    //   every particle is slower than sleepSpeed, and
    //   its constraints are within sleepError, or their error has stopped changing
    //   (|change per frame| < sleepErrorDrift) - the equilibrium case, e.g. a cloth sagging under gravity.
    float sleepSpeed = 0.002f;      // m/s
    int sleepFrames = 30;           // consecutive calm frames before a block sleeps
    float sleepError = 2e-3f;       // relative constraint error regarded as "flat"
    float sleepErrorDrift = 5e-6f;  // relative error change per frame regarded as "settled"
    // When more than this share of blocks is awake, scheduling only adds overhead, so the
    // active solver runs the plain global step for that frame (1 = never fall back).
    float globalFallback = 0.7f;
};

struct StepStats
{
    int iterations = 0;    // Jacobi iterations actually run (summed over substeps)
    float maxError = 0.0f; // last measured max |C|/rest over solved constraints
    double stepMs = 0.0;   // wall time of the whole step
    double solveMs = 0.0;  // time spent in the constraint iterations
    int activeBlocks = 0;  // active-block solver only
    int totalBlocks = 0;
    int solvedConstraints = 0; // constraints projected per iteration
    bool globalFallback = false; // active solver ran the global path this frame
};

enum SolverType
{
    CPU_GLOBAL,
    CPU_ACTIVE,
    GPU_GLOBAL,
    GPU_ACTIVE
};
struct GpuBuffers;

class Solver
{
  public:
    Solver() = default;
    ~Solver();
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    bool Init(SolverType type, const SolverConfig& settings, const Cloth& initialCloth);
    StepStats Step(float frameDt);
    void SetGrab(int index, Vec3 position);
    void GetCloth(Cloth& result) const;
    std::vector<unsigned char> GetActiveBlocks() const;
    int GetBlockSize() const;
    const char* GetName() const;

  private:
    SolverType mode = CPU_GLOBAL;
    SolverConfig config;
    int grabIndex = -1;
    Vec3 grabPosition = {0, 0, 0};
    bool firstFrame = true;

    Cloth cloth;
    std::vector<Vec3> predicted;
    std::vector<Vec3> correction;
    std::vector<int> correctionCount;
    BlockGrid blockGrid;
    BlockConstraintTable blockTable;
    std::vector<unsigned char> activeBlocks;
    std::vector<float> blockError, blockErrorPrev; // max |C|/rest per block, this and previous frame
    std::vector<int> calmFrames;                  // consecutive calm frames per block
    StepStats StepCpuGlobal(float frameDt);
    StepStats StepCpuActive(float frameDt);

    GpuBuffers* gpu = nullptr;
    void InitGpu(const Cloth& initialCloth);
    void FreeGpu();
    void GetGpuCloth(Cloth& result) const;
    std::vector<unsigned char> GetGpuActiveBlocks() const;
    StepStats StepGpuGlobal(float frameDt);
    StepStats StepGpuActive(float frameDt);
};

#if defined(PBD_WITH_CUDA)
bool cudaDeviceAvailable();
#endif
} // namespace pbd
