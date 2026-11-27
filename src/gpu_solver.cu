#include <algorithm>
#include <cuda_runtime.h>

#include "pbd/timer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pbd/blocks.h"
#include "pbd/solver.h"

#define CUDA_CHECK(expr)                                                                                          \
    do                                                                                                            \
    {                                                                                                             \
        cudaError_t err__ = (expr);                                                                               \
        if (err__ != cudaSuccess)                                                                                 \
        {                                                                                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__);      \
            std::exit(1);                                                                                         \
        }                                                                                                         \
    } while (0)

namespace pbd
{

constexpr int THREADS_PER_BLOCK = 256;
inline int BlockCountFor(int n)
{
    return (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

__device__ __forceinline__ void atomicAddVec(Vec3* a, Vec3 b)
{
    atomicAdd(&a->x, b.x);
    atomicAdd(&a->y, b.y);
    atomicAdd(&a->z, b.z);
}

// Non-negative floats keep their order when compared as unsigned ints.
__device__ __forceinline__ void atomicMaxFloat(unsigned* addr, float v)
{
    atomicMax(addr, __float_as_uint(v));
}

// 전체 계산: 입자 또는 제약 하나당 스레드 하나를 사용한다.
__global__ void PredictPositions(int n, const Vec3* x, Vec3* velocity, Vec3* predicted, const float* invMass,
                                 Vec3 g, float dt, float damping, int grabIndex, Vec3 target)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    if (i == grabIndex)
    {
        predicted[i] = target;
        return;
    }
    if (invMass[i] == 0.0f)
    {
        predicted[i] = x[i];
        return;
    }
    const Vec3 vi = (velocity[i] + g * dt) * (1.0f - damping);
    velocity[i] = vi;
    predicted[i] = x[i] + vi * dt;
}

__global__ void SolveConstraints(int m, const DistanceConstraint* c, const Vec3* predicted, const float* invMass,
                                 int grabIndex, Vec3* correction, int* count)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= m)
    {
        return;
    }
    const DistanceConstraint constraint = c[k];
    const float invMassA = constraint.i == grabIndex ? 0.0f : invMass[constraint.i];
    const float invMassB = constraint.j == grabIndex ? 0.0f : invMass[constraint.j];
    Vec3 correctionA, correctionB;
    if (!project_distance(predicted[constraint.i], predicted[constraint.j], invMassA, invMassB, constraint.rest,
                          constraint.stiffness, correctionA, correctionB))
    {
        return;
    }
    if (invMassA > 0.0f)
    {
        atomicAddVec(&correction[constraint.i], correctionA);
        atomicAdd(&count[constraint.i], 1);
    }
    if (invMassB > 0.0f)
    {
        atomicAddVec(&correction[constraint.j], correctionB);
        atomicAdd(&count[constraint.j], 1);
    }
}

__global__ void ApplyCorrections(int n, Vec3* predicted, Vec3* correction, int* count, float relax)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const int cnt = count[i];
    if (cnt == 0)
    {
        return;
    }
    predicted[i] += correction[i] * (relax / static_cast<float>(cnt));
    correction[i] = Vec3{0, 0, 0}; // reset for the next iteration, no memset needed
    count[i] = 0;
}

__global__ void CheckError(int m, const DistanceConstraint* c, const Vec3* predicted, unsigned* errBits)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= m)
    {
        return;
    }
    const DistanceConstraint constraint = c[k];
    atomicMaxFloat(errBits, relative_error(predicted[constraint.i], predicted[constraint.j], constraint.rest));
}

__global__ void UpdateVelocities(int n, Vec3* x, Vec3* velocity, const Vec3* predicted, const float* invMass,
                                 float invDt, int grabIndex)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    velocity[i] = (invMass[i] != 0.0f && i != grabIndex) ? (predicted[i] - x[i]) * invDt : Vec3{0, 0, 0};
    x[i] = predicted[i];
}

// 활성 블록 계산: 블록 번호로 이번 프레임의 입자를 찾는다.
struct GridInfo
{
    int nx, ny, B, bx, by;
    __device__ __forceinline__ int blockOfParticle(int p) const
    {
        return ((p / nx) / B) * bx + (p % nx) / B;
    }
    // Particle handled by thread `t` of active block `b`; -1 when outside the cloth.
    __device__ __forceinline__ int particleOf(int b, int t) const
    {
        const int i = (b % bx) * B + t % B, j = (b / bx) * B + t / B;
        return (i < nx && j < ny) ? j * nx + i : -1;
    }
};

__global__ void MarkMovingBlocks(int n, GridInfo g, const Vec3* velocity, const float* invMass, float sleepSpeed,
                                 int grabIndex, unsigned int* seed)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const bool moving = invMass[i] > 0.0f && length(velocity[i]) > sleepSpeed;
    if (moving || i == grabIndex)
    {
        atomicExch(&seed[g.blockOfParticle(i)], 1u);
    }
}

__global__ void MarkNeighborBlocks(GridInfo g, const unsigned int* seed, unsigned char* active, int* list,
                                   int* numActive)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= g.bx * g.by)
    {
        return;
    }
    const int x = b % g.bx, y = b / g.bx;
    unsigned char on = 0;
    for (int dy = -1; dy <= 1 && !on; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int X = x + dx, Y = y + dy;
            if (X >= 0 && Y >= 0 && X < g.bx && Y < g.by && seed[Y * g.bx + X])
            {
                on = 1;
                break;
            }
        }
    }
    active[b] = on;
    if (on)
    {
        list[atomicAdd(numActive, 1)] = b;
    }
}

// A block stays awake until it has been calm for `sleepFrames` frames in a row.
// Calm = no fast particle (restless == 0) and its constraints are flat or no longer changing.
__global__ void UpdateCalmBlocks(int blockCount, const unsigned int* restless, const unsigned* error,
                                 const unsigned* previous, int* calm, int sleepFrames, float sleepError,
                                 float sleepErrorDrift, unsigned int* seed)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blockCount)
    {
        return;
    }
    const float e = __uint_as_float(error[b]);
    const float p = __uint_as_float(previous[b]);
    const bool stillMoving = restless[b] != 0u || (e > sleepError && fabsf(e - p) > sleepErrorDrift);
    const int c = stillMoving ? 0 : min(calm[b] + 1, 1 << 20);
    calm[b] = c;
    seed[b] = c < sleepFrames ? 1u : 0u;
}

// previous = error, error = 0 for the listed blocks (all blocks when `blocks` is null).
__global__ void ResetBlockErrors(int count, const int* blocks, unsigned* error, unsigned* previous)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count)
    {
        return;
    }
    const int b = blocks ? blocks[t] : t;
    previous[b] = error[b];
    error[b] = 0u;
}

// Max relative error per block over the listed constraints (all constraints when `list` is null).
__global__ void AccumulateBlockErrors(int count, const int* list, GridInfo g, const unsigned char* active,
                                      const DistanceConstraint* c, const Vec3* x, unsigned* error)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count)
    {
        return;
    }
    const DistanceConstraint ck = c[list ? list[k] : k];
    const float e = relative_error(x[ck.i], x[ck.j], ck.rest);
    const int bi = g.blockOfParticle(ck.i), bj = g.blockOfParticle(ck.j);
    if (!active || active[bi])
    {
        atomicMaxFloat(&error[bi], e);
    }
    if (bj != bi && (!active || active[bj]))
    {
        atomicMaxFloat(&error[bj], e);
    }
}

// One CUDA block per active particle-block; threads stride over its CSR range.
__global__ void CollectActiveConstraints(const int* activeList, const int* start, const int* table,
                                         unsigned* stamp, unsigned tag, int* outList, int* outCount)
{
    const int b = activeList[blockIdx.x];
    for (int k = start[b] + threadIdx.x; k < start[b + 1]; k += blockDim.x)
    {
        const int c = table[k];
        if (atomicExch(&stamp[c], tag) != tag)
        {
            outList[atomicAdd(outCount, 1)] = c;
        }
    }
}

__global__ void PredictActivePositions(GridInfo g, const int* activeList, const Vec3* x, Vec3* velocity,
                                       Vec3* predicted, const float* invMass, Vec3 grav, float dt, float damping,
                                       int grabIndex, Vec3 target)
{
    const int i = g.particleOf(activeList[blockIdx.x], threadIdx.x);
    if (i < 0)
    {
        return;
    }
    if (i == grabIndex)
    {
        predicted[i] = target;
        return;
    }
    if (invMass[i] == 0.0f)
    {
        predicted[i] = x[i];
        return;
    }
    const Vec3 vi = (velocity[i] + grav * dt) * (1.0f - damping);
    velocity[i] = vi;
    predicted[i] = x[i] + vi * dt;
}

__global__ void SolveActiveConstraints(int count, const int* list, GridInfo g, const unsigned char* active,
                                       const DistanceConstraint* c, const Vec3* predicted, const float* invMass,
                                       int grabIndex, Vec3* correction, int* cnt)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count)
    {
        return;
    }
    const DistanceConstraint constraint = c[list[t]];
    // Sleeping particles behave as anchors for the boundary of the active region.
    const float invMassA =
        (constraint.i == grabIndex || !active[g.blockOfParticle(constraint.i)]) ? 0.0f : invMass[constraint.i];
    const float invMassB =
        (constraint.j == grabIndex || !active[g.blockOfParticle(constraint.j)]) ? 0.0f : invMass[constraint.j];
    Vec3 correctionA, correctionB;
    if (!project_distance(predicted[constraint.i], predicted[constraint.j], invMassA, invMassB, constraint.rest,
                          constraint.stiffness, correctionA, correctionB))
    {
        return;
    }
    if (invMassA > 0.0f)
    {
        atomicAddVec(&correction[constraint.i], correctionA);
        atomicAdd(&cnt[constraint.i], 1);
    }
    if (invMassB > 0.0f)
    {
        atomicAddVec(&correction[constraint.j], correctionB);
        atomicAdd(&cnt[constraint.j], 1);
    }
}

__global__ void ApplyActiveCorrections(GridInfo g, const int* activeList, Vec3* predicted, Vec3* correction,
                                       int* cnt, float relax)
{
    const int i = g.particleOf(activeList[blockIdx.x], threadIdx.x);
    if (i < 0)
    {
        return;
    }
    const int n = cnt[i];
    if (n == 0)
    {
        return;
    }
    predicted[i] += correction[i] * (relax / static_cast<float>(n));
    correction[i] = Vec3{0, 0, 0};
    cnt[i] = 0;
}

__global__ void CheckActiveError(int count, const int* list, const DistanceConstraint* c, const Vec3* predicted,
                                 unsigned* errBits)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count)
    {
        return;
    }
    const DistanceConstraint constraint = c[list[t]];
    atomicMaxFloat(errBits, relative_error(predicted[constraint.i], predicted[constraint.j], constraint.rest));
}

__global__ void UpdateActiveVelocities(GridInfo g, const int* activeList, Vec3* x, Vec3* velocity,
                                       const Vec3* predicted, const float* invMass, float invDt, int grabIndex)
{
    const int i = g.particleOf(activeList[blockIdx.x], threadIdx.x);
    if (i < 0)
    {
        return;
    }
    velocity[i] = (invMass[i] != 0.0f && i != grabIndex) ? (predicted[i] - x[i]) * invDt : Vec3{0, 0, 0};
    x[i] = predicted[i];
}

struct GpuBuffers
{
    Vec3* dPosition = nullptr;
    Vec3* dVelocity = nullptr;
    Vec3* dPredicted = nullptr;
    float* dInvMass = nullptr;
    DistanceConstraint* dConstraints = nullptr;
    Vec3* dCorrection = nullptr;
    int* dCorrectionCount = nullptr;
    unsigned* dError = nullptr;
    int* dBlockStart = nullptr;
    int* dBlockConstraints = nullptr;
    unsigned* dVisited = nullptr;
    int* dConstraintList = nullptr;
    unsigned int* dSeed = nullptr;
    unsigned char* dActive = nullptr;
    int* dActiveList = nullptr;
    int* dCounters = nullptr;
    unsigned int* dRestless = nullptr;
    int* dCalm = nullptr;
    unsigned* dBlockError = nullptr;
    unsigned* dBlockErrorPrev = nullptr;

    int particleCount = 0;
    int constraintCount = 0;
    int blockCount = 0;
    unsigned frameTag = 0;
    GridInfo grid = {};
    cudaEvent_t solveStart, solveEnd;
};

void Solver::InitGpu(const Cloth& initialCloth)
{
    gpu = new GpuBuffers;
    gpu->particleCount = initialCloth.particleCount();
    gpu->constraintCount = initialCloth.constraintCount();
    CUDA_CHECK(cudaEventCreate(&gpu->solveStart));
    CUDA_CHECK(cudaEventCreate(&gpu->solveEnd));
    CUDA_CHECK(cudaMalloc(&gpu->dPosition, gpu->particleCount * sizeof(Vec3)));
    CUDA_CHECK(cudaMemcpy(gpu->dPosition, initialCloth.position.data(), gpu->particleCount * sizeof(Vec3),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&gpu->dVelocity, gpu->particleCount * sizeof(Vec3)));
    CUDA_CHECK(cudaMemcpy(gpu->dVelocity, initialCloth.velocity.data(), gpu->particleCount * sizeof(Vec3),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&gpu->dPredicted, gpu->particleCount * sizeof(Vec3)));
    CUDA_CHECK(cudaMemcpy(gpu->dPredicted, initialCloth.position.data(), gpu->particleCount * sizeof(Vec3),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&gpu->dInvMass, gpu->particleCount * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(gpu->dInvMass, initialCloth.invMass.data(), gpu->particleCount * sizeof(float),
                          cudaMemcpyHostToDevice));
    int constraintStorage = std::max(1, gpu->constraintCount);
    CUDA_CHECK(cudaMalloc(&gpu->dConstraints, constraintStorage * sizeof(DistanceConstraint)));
    if (gpu->constraintCount > 0)
    {
        CUDA_CHECK(cudaMemcpy(gpu->dConstraints, initialCloth.constraints.data(),
                              gpu->constraintCount * sizeof(DistanceConstraint), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&gpu->dCorrection, gpu->particleCount * sizeof(Vec3)));
    CUDA_CHECK(cudaMemset(gpu->dCorrection, 0, gpu->particleCount * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&gpu->dCorrectionCount, gpu->particleCount * sizeof(int)));
    CUDA_CHECK(cudaMemset(gpu->dCorrectionCount, 0, gpu->particleCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu->dError, 1 * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(gpu->dError, 0, 1 * sizeof(unsigned)));

    if (mode != GPU_ACTIVE)
    {
        return;
    }
    BlockGrid blocks = BlockGrid::make(initialCloth.nx, initialCloth.ny, config.blockSize);
    BlockConstraintTable table = BlockConstraintTable::build(initialCloth, blocks);
    gpu->grid = GridInfo{initialCloth.nx, initialCloth.ny, blocks.blockSize, blocks.bx, blocks.by};
    gpu->blockCount = blocks.count();
    CUDA_CHECK(cudaMalloc(&gpu->dBlockStart, table.start.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu->dBlockStart, table.start.data(), table.start.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    if (!table.constraints.empty())
    {
        CUDA_CHECK(cudaMalloc(&gpu->dBlockConstraints, table.constraints.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(gpu->dBlockConstraints, table.constraints.data(),
                              table.constraints.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&gpu->dVisited, constraintStorage * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(gpu->dVisited, 0, constraintStorage * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&gpu->dConstraintList, constraintStorage * sizeof(int)));
    CUDA_CHECK(cudaMemset(gpu->dConstraintList, 0, constraintStorage * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu->dSeed, gpu->blockCount * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(gpu->dSeed, 0, gpu->blockCount * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&gpu->dActive, gpu->blockCount * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(gpu->dActive, 0, gpu->blockCount * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&gpu->dActiveList, gpu->blockCount * sizeof(int)));
    CUDA_CHECK(cudaMemset(gpu->dActiveList, 0, gpu->blockCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu->dCounters, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(gpu->dCounters, 0, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu->dRestless, gpu->blockCount * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&gpu->dCalm, gpu->blockCount * sizeof(int)));
    CUDA_CHECK(cudaMemset(gpu->dCalm, 0, gpu->blockCount * sizeof(int))); // every block starts awake
    CUDA_CHECK(cudaMalloc(&gpu->dBlockError, gpu->blockCount * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(gpu->dBlockError, 0, gpu->blockCount * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&gpu->dBlockErrorPrev, gpu->blockCount * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(gpu->dBlockErrorPrev, 0, gpu->blockCount * sizeof(unsigned)));
}
void Solver::FreeGpu()
{
    if (!gpu)
    {
        return;
    }
    cudaFree(gpu->dPosition);
    cudaFree(gpu->dVelocity);
    cudaFree(gpu->dPredicted);
    cudaFree(gpu->dInvMass);
    cudaFree(gpu->dConstraints);
    cudaFree(gpu->dCorrection);
    cudaFree(gpu->dCorrectionCount);
    cudaFree(gpu->dError);
    cudaFree(gpu->dBlockStart);
    cudaFree(gpu->dBlockConstraints);
    cudaFree(gpu->dVisited);
    cudaFree(gpu->dConstraintList);
    cudaFree(gpu->dSeed);
    cudaFree(gpu->dActive);
    cudaFree(gpu->dActiveList);
    cudaFree(gpu->dCounters);
    cudaFree(gpu->dRestless);
    cudaFree(gpu->dCalm);
    cudaFree(gpu->dBlockError);
    cudaFree(gpu->dBlockErrorPrev);
    cudaEventDestroy(gpu->solveStart);
    cudaEventDestroy(gpu->solveEnd);
    delete gpu;
    gpu = nullptr;
}

void Solver::GetGpuCloth(Cloth& result) const
{
    result.position.resize(gpu->particleCount);
    result.velocity.resize(gpu->particleCount);
    CUDA_CHECK(cudaMemcpy(result.position.data(), gpu->dPosition, gpu->particleCount * sizeof(Vec3),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.velocity.data(), gpu->dVelocity, gpu->particleCount * sizeof(Vec3),
                          cudaMemcpyDeviceToHost));
}
std::vector<unsigned char> Solver::GetGpuActiveBlocks() const
{
    std::vector<unsigned char> mask(gpu->blockCount);
    CUDA_CHECK(cudaMemcpy(mask.data(), gpu->dActive, gpu->blockCount, cudaMemcpyDeviceToHost));
    return mask;
}
static float ReadAndResetError(unsigned* deviceError)
{
    unsigned bits = 0;
    CUDA_CHECK(cudaMemcpy(&bits, deviceError, sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(deviceError, 0, sizeof(unsigned)));
    float e;
    std::memcpy(&e, &bits, sizeof(float));
    return e;
}
StepStats Solver::StepGpuGlobal(float frameDt)
{
    const ClockTime stepStart = SimulationClock::now();
    StepStats stats;
    const int substeps = config.substeps > 0 ? config.substeps : 1;
    const float dt = frameDt / static_cast<float>(substeps);
    const float damping = config.damping / static_cast<float>(substeps);
    float solveMs = 0.0f;

    for (int k = 0; k < substeps; ++k)
    {
        PredictPositions<<<BlockCountFor(gpu->particleCount), THREADS_PER_BLOCK>>>(
            gpu->particleCount, gpu->dPosition, gpu->dVelocity, gpu->dPredicted, gpu->dInvMass, config.gravity, dt,
            damping, grabIndex, grabPosition);
        CUDA_CHECK(cudaEventRecord(gpu->solveStart));
        for (int it = 0; it < config.maxIterations; ++it)
        {
            SolveConstraints<<<BlockCountFor(gpu->constraintCount), THREADS_PER_BLOCK>>>(
                gpu->constraintCount, gpu->dConstraints, gpu->dPredicted, gpu->dInvMass, grabIndex,
                gpu->dCorrection, gpu->dCorrectionCount);
            ApplyCorrections<<<BlockCountFor(gpu->particleCount), THREADS_PER_BLOCK>>>(
                gpu->particleCount, gpu->dPredicted, gpu->dCorrection, gpu->dCorrectionCount, config.relaxation);
            stats.iterations++;
            if ((it + 1) % config.checkEvery == 0 || it + 1 == config.maxIterations)
            {
                CheckError<<<BlockCountFor(gpu->constraintCount), THREADS_PER_BLOCK>>>(
                    gpu->constraintCount, gpu->dConstraints, gpu->dPredicted, gpu->dError);
                stats.maxError = ReadAndResetError(gpu->dError); // synchronises: the only host round trip
                if (stats.maxError < config.tolerance)
                {
                    break;
                }
            }
        }
        CUDA_CHECK(cudaEventRecord(gpu->solveEnd));
        UpdateVelocities<<<BlockCountFor(gpu->particleCount), THREADS_PER_BLOCK>>>(
            gpu->particleCount, gpu->dPosition, gpu->dVelocity, gpu->dPredicted, gpu->dInvMass, 1.0f / dt,
            grabIndex);
        CUDA_CHECK(cudaEventSynchronize(gpu->solveEnd));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, gpu->solveStart, gpu->solveEnd));
        solveMs += ms;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    stats.solveMs = solveMs;
    stats.solvedConstraints = gpu->constraintCount;
    stats.stepMs = ElapsedMs(stepStart);
    return stats;
}

StepStats Solver::StepGpuActive(float frameDt)
{
    const ClockTime stepStart = SimulationClock::now();
    StepStats stats;
    stats.totalBlocks = gpu->blockCount;
    const int threadsPerBlock = gpu->grid.B * gpu->grid.B;

    // 속도가 기준을 넘는 블록과 잡고 있는 입자의 블록을 표시한다.
    CUDA_CHECK(cudaMemset(gpu->dRestless, 0, gpu->blockCount * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(gpu->dCounters, 0, sizeof(int) * 2));
    firstFrame = false;
    MarkMovingBlocks<<<BlockCountFor(gpu->particleCount), THREADS_PER_BLOCK>>>(
        gpu->particleCount, gpu->grid, gpu->dVelocity, gpu->dInvMass, config.sleepSpeed, grabIndex, gpu->dRestless);
    // sleepFrames 동안 조용하고 제약이 평평하거나 더 이상 변하지 않는 블록만 잠든다.
    UpdateCalmBlocks<<<BlockCountFor(gpu->blockCount), THREADS_PER_BLOCK>>>(
        gpu->blockCount, gpu->dRestless, gpu->dBlockError, gpu->dBlockErrorPrev, gpu->dCalm, config.sleepFrames,
        config.sleepError, config.sleepErrorDrift, gpu->dSeed);
    // 보정이 경계에서 끊기지 않도록 이웃 블록까지 깨운다.
    MarkNeighborBlocks<<<BlockCountFor(gpu->blockCount), THREADS_PER_BLOCK>>>(
        gpu->grid, gpu->dSeed, gpu->dActive, gpu->dActiveList, &gpu->dCounters[0]);
    int activeBlockCount = 0;
    CUDA_CHECK(cudaMemcpy(&activeBlockCount, &gpu->dCounters[0], sizeof(int), cudaMemcpyDeviceToHost));
    stats.activeBlocks = activeBlockCount;
    if (activeBlockCount == 0)
    { // everything is asleep
        stats.stepMs = ElapsedMs(stepStart);
        return stats;
    }
    // 대부분의 블록이 깨어 있으면 목록을 만드는 비용만 늘어나므로 전체 계산으로 전환한다.
    if (activeBlockCount > config.globalFallback * gpu->blockCount)
    {
        StepStats global = StepGpuGlobal(frameDt);
        CUDA_CHECK(cudaMemset(gpu->dActive, 1, gpu->blockCount * sizeof(unsigned char)));
        ResetBlockErrors<<<BlockCountFor(gpu->blockCount), THREADS_PER_BLOCK>>>(
            gpu->blockCount, nullptr, gpu->dBlockError, gpu->dBlockErrorPrev);
        AccumulateBlockErrors<<<BlockCountFor(gpu->constraintCount), THREADS_PER_BLOCK>>>(
            gpu->constraintCount, nullptr, gpu->grid, nullptr, gpu->dConstraints, gpu->dPosition, gpu->dBlockError);
        CUDA_CHECK(cudaDeviceSynchronize());
        global.activeBlocks = gpu->blockCount;
        global.totalBlocks = gpu->blockCount;
        global.globalFallback = true;
        global.stepMs = ElapsedMs(stepStart);
        return global;
    }

    ++gpu->frameTag;
    // 두 블록에 걸친 제약은 목록에 한 번만 넣는다.
    CollectActiveConstraints<<<activeBlockCount, 128>>>(gpu->dActiveList, gpu->dBlockStart, gpu->dBlockConstraints,
                                                        gpu->dVisited, gpu->frameTag, gpu->dConstraintList,
                                                        &gpu->dCounters[1]);
    int activeConstraintCount = 0;
    CUDA_CHECK(cudaMemcpy(&activeConstraintCount, &gpu->dCounters[1], sizeof(int), cudaMemcpyDeviceToHost));
    stats.solvedConstraints = activeConstraintCount;

    // 한 프레임 안에서는 같은 목록을 모든 서브스텝에서 재사용한다.
    const int substeps = config.substeps > 0 ? config.substeps : 1;
    const float dt = frameDt / static_cast<float>(substeps);
    const float damping = config.damping / static_cast<float>(substeps);
    float solveMs = 0.0f;
    for (int k = 0; k < substeps; ++k)
    {
        PredictActivePositions<<<activeBlockCount, threadsPerBlock>>>(
            gpu->grid, gpu->dActiveList, gpu->dPosition, gpu->dVelocity, gpu->dPredicted, gpu->dInvMass,
            config.gravity, dt, damping, grabIndex, grabPosition);
        CUDA_CHECK(cudaEventRecord(gpu->solveStart));
        for (int it = 0; it < config.maxIterations && activeConstraintCount > 0; ++it)
        {
            SolveActiveConstraints<<<BlockCountFor(activeConstraintCount), THREADS_PER_BLOCK>>>(
                activeConstraintCount, gpu->dConstraintList, gpu->grid, gpu->dActive, gpu->dConstraints,
                gpu->dPredicted, gpu->dInvMass, grabIndex, gpu->dCorrection, gpu->dCorrectionCount);
            ApplyActiveCorrections<<<activeBlockCount, threadsPerBlock>>>(
                gpu->grid, gpu->dActiveList, gpu->dPredicted, gpu->dCorrection, gpu->dCorrectionCount,
                config.relaxation);
            stats.iterations++;
            if ((it + 1) % config.checkEvery == 0 || it + 1 == config.maxIterations)
            {
                CheckActiveError<<<BlockCountFor(activeConstraintCount), THREADS_PER_BLOCK>>>(
                    activeConstraintCount, gpu->dConstraintList, gpu->dConstraints, gpu->dPredicted, gpu->dError);
                stats.maxError = ReadAndResetError(gpu->dError);
                if (stats.maxError < config.tolerance)
                {
                    break;
                }
            }
        }
        CUDA_CHECK(cudaEventRecord(gpu->solveEnd));
        UpdateActiveVelocities<<<activeBlockCount, threadsPerBlock>>>(gpu->grid, gpu->dActiveList, gpu->dPosition,
                                                                      gpu->dVelocity, gpu->dPredicted,
                                                                      gpu->dInvMass, 1.0f / dt, grabIndex);
        CUDA_CHECK(cudaEventSynchronize(gpu->solveEnd));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, gpu->solveStart, gpu->solveEnd));
        solveMs += ms;
    }
    // 이번 프레임에 계산한 블록의 제약 오차를 갱신한다 (잠든 블록은 값이 그대로 유지된다).
    ResetBlockErrors<<<BlockCountFor(activeBlockCount), THREADS_PER_BLOCK>>>(
        activeBlockCount, gpu->dActiveList, gpu->dBlockError, gpu->dBlockErrorPrev);
    if (activeConstraintCount > 0)
    {
        AccumulateBlockErrors<<<BlockCountFor(activeConstraintCount), THREADS_PER_BLOCK>>>(
            activeConstraintCount, gpu->dConstraintList, gpu->grid, gpu->dActive, gpu->dConstraints,
            gpu->dPosition, gpu->dBlockError);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    stats.solveMs = solveMs;
    stats.stepMs = ElapsedMs(stepStart);
    return stats;
}


bool cudaDeviceAvailable()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}
}
