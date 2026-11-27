#include <algorithm>
#include "pbd/timer.h"
#include <cstdio>
#include "pbd/solver.h"

namespace pbd
{

Solver::~Solver()
{
#ifdef PBD_WITH_CUDA
    FreeGpu();
#endif
}

bool Solver::Init(SolverType type, const SolverConfig& settings, const Cloth& initialCloth)
{
    if ((type == CPU_ACTIVE || type == GPU_ACTIVE) && settings.blockSize < 1)
    {
        return false;
    }
    if (type == GPU_ACTIVE && settings.blockSize > 32)
    {
        return false;
    }
#ifdef PBD_WITH_CUDA
    FreeGpu();
#else
    if (type == GPU_GLOBAL || type == GPU_ACTIVE)
    {
        return false;
    }
#endif
    mode = type;
    config = settings;
    grabIndex = -1;
    grabPosition = {0, 0, 0};
    firstFrame = true;
    activeBlocks.clear();
#ifdef PBD_WITH_CUDA
    if (mode == GPU_GLOBAL || mode == GPU_ACTIVE)
    {
        InitGpu(initialCloth);
        return true;
    }
#endif
    cloth = initialCloth;
    predicted = cloth.position;
    correction.assign(cloth.particleCount(), Vec3{0, 0, 0});
    correctionCount.assign(cloth.particleCount(), 0);
    if (mode == CPU_ACTIVE)
    {
        blockGrid = BlockGrid::make(cloth.nx, cloth.ny, config.blockSize);
        blockTable = BlockConstraintTable::build(cloth, blockGrid);
        blockError.assign(blockGrid.count(), 0.0f);
        blockErrorPrev.assign(blockGrid.count(), 0.0f);
        calmFrames.assign(blockGrid.count(), 0); // every block starts awake
    }
    return true;
}

void Solver::SetGrab(int index, Vec3 position)
{
    grabIndex = index;
    grabPosition = position;
}

StepStats Solver::Step(float frameDt)
{
    switch (mode)
    {
    case CPU_GLOBAL:
        return StepCpuGlobal(frameDt);
    case CPU_ACTIVE:
        return StepCpuActive(frameDt);
#ifdef PBD_WITH_CUDA
    case GPU_GLOBAL:
        return StepGpuGlobal(frameDt);
    case GPU_ACTIVE:
        return StepGpuActive(frameDt);
#endif
    default:
        return StepStats();
    }
}

const char* Solver::GetName() const
{
    switch (mode)
    {
    case CPU_GLOBAL:
        return "cpu";
    case CPU_ACTIVE:
        return "cpu-active";
    case GPU_GLOBAL:
        return "gpu-global";
    case GPU_ACTIVE:
        return "gpu-active";
    }
    return "unknown";
}

void Solver::GetCloth(Cloth& result) const
{
#ifdef PBD_WITH_CUDA
    if (gpu)
    {
        GetGpuCloth(result);
        return;
    }
#endif
    result.position = cloth.position;
    result.velocity = cloth.velocity;
}

int Solver::GetBlockSize() const
{
    if (mode == CPU_ACTIVE || mode == GPU_ACTIVE)
    {
        return config.blockSize;
    }
    return 0;
}

std::vector<unsigned char> Solver::GetActiveBlocks() const
{
#ifdef PBD_WITH_CUDA
    if (mode == GPU_ACTIVE)
    {
        return GetGpuActiveBlocks();
    }
#endif
    return activeBlocks;
}

StepStats Solver::StepCpuGlobal(float frameDt)
{

    const ClockTime stepStart = SimulationClock::now();
    StepStats stats;
    const int particleCount = static_cast<int>(cloth.position.size());
    const int substeps = std::max(1, config.substeps);
    const float dt = frameDt / static_cast<float>(substeps);
    const float damping = config.damping / static_cast<float>(substeps);
    double solveMs = 0.0;

    for (int substep = 0; substep < substeps; substep++)
    {
        // 속도와 중력으로 다음 위치를 예측한다.
        for (int i = 0; i < particleCount; ++i)
        {
            if (i == grabIndex)
            {
                predicted[i] = grabPosition;
                continue;
            }
            if (cloth.invMass[i] == 0.0f)
            {
                predicted[i] = cloth.position[i];
                continue;
            }
            cloth.velocity[i] = (cloth.velocity[i] + config.gravity * dt) * (1.0f - damping);
            predicted[i] = cloth.position[i] + cloth.velocity[i] * dt;
        }

        // 같은 예측 위치에서 거리 제약을 계산한 뒤 입자별 평균 보정을 적용한다.
        const ClockTime solveStart = SimulationClock::now();
        for (int it = 0; it < config.maxIterations; ++it)
        {
            std::fill(correction.begin(), correction.end(), Vec3{0, 0, 0});
            std::fill(correctionCount.begin(), correctionCount.end(), 0);
            for (int k = 0; k < cloth.constraintCount(); k++)
            {
                const DistanceConstraint& c = cloth.constraints[k];
                const float wi = (c.i == grabIndex) ? 0.0f : cloth.invMass[c.i];
                const float wj = (c.j == grabIndex) ? 0.0f : cloth.invMass[c.j];
                Vec3 dpi, dpj;
                if (!project_distance(predicted[c.i], predicted[c.j], wi, wj, c.rest, c.stiffness, dpi, dpj))
                {
                    continue;
                }
                if (wi > 0.0f)
                {
                    correction[c.i] += dpi;
                    correctionCount[c.i]++;
                }
                if (wj > 0.0f)
                {
                    correction[c.j] += dpj;
                    correctionCount[c.j]++;
                }
            }
            for (int i = 0; i < particleCount; ++i)
            {
                if (correctionCount[i])
                {
                    predicted[i] += correction[i] * (config.relaxation / static_cast<float>(correctionCount[i]));
                }
            }
            stats.iterations++;

            if ((it + 1) % config.checkEvery == 0 || it + 1 == config.maxIterations)
            {
                float e = 0.0f;
                for (int k = 0; k < cloth.constraintCount(); k++)
                {
                    const DistanceConstraint& c = cloth.constraints[k];
                    e = std::max(e, relative_error(predicted[c.i], predicted[c.j], c.rest));
                }
                stats.maxError = e;
                if (e < config.tolerance)
                {
                    break;
                }
            }
        }
        solveMs += ElapsedMs(solveStart);

        // 보정 전후의 위치 차이로 속도를 갱신한다.
        const float inv = 1.0f / dt;
        for (int i = 0; i < particleCount; ++i)
        {
            cloth.velocity[i] = (cloth.invMass[i] != 0.0f && i != grabIndex)
                                    ? (predicted[i] - cloth.position[i]) * inv
                                    : Vec3{0, 0, 0};
            cloth.position[i] = predicted[i];
        }
    }

    stats.solvedConstraints = static_cast<int>(cloth.constraints.size());
    stats.solveMs = solveMs;
    stats.stepMs = ElapsedMs(stepStart);
    return stats;
}
}
