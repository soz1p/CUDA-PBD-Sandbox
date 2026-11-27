#include <algorithm>
#include <cmath>
#include "pbd/timer.h"
#include <cstdio>
#include "pbd/solver.h"

namespace pbd
{

// Max relative error per block over the given constraints; only blocks marked in `solved` are refreshed.
static void UpdateBlockErrors(const Cloth& cloth, const BlockGrid& grid, const std::vector<int>* list,
                              const std::vector<unsigned char>& solved, std::vector<float>& error,
                              std::vector<float>& previous)
{
    for (size_t b = 0; b < error.size(); ++b)
    {
        if (solved[b])
        {
            previous[b] = error[b];
            error[b] = 0.0f;
        }
    }
    const int count = list ? static_cast<int>(list->size()) : cloth.constraintCount();
    for (int k = 0; k < count; ++k)
    {
        const DistanceConstraint& ck = cloth.constraints[list ? (*list)[k] : k];
        const float e = relative_error(cloth.position[ck.i], cloth.position[ck.j], ck.rest);
        for (int p : {ck.i, ck.j})
        {
            const int b = grid.blockOf(p % cloth.nx, p / cloth.nx);
            if (solved[b])
            {
                error[b] = std::max(error[b], e);
            }
        }
    }
}

StepStats Solver::StepCpuActive(float frameDt)
{

    const ClockTime stepStart = SimulationClock::now();
    StepStats stats;
    stats.totalBlocks = blockGrid.count();
    std::vector<Vec3>& x = cloth.position;
    std::vector<Vec3>& v = cloth.velocity;
    const std::vector<float>& w = cloth.invMass;
    const int nx = cloth.nx;

    // 1. 블록마다 아직 움직이는지(restless) 판단한다.
    //    - 입자 속도가 sleepSpeed를 넘거나 잡힌 입자가 있으면 움직이는 블록
    //    - 제약 오차가 sleepError보다 크고 아직 변하고 있으면(평형 전) 움직이는 블록
    const int blockCount = blockGrid.count();
    std::vector<unsigned char> restless(blockCount, 0);
    for (int i = 0; i < cloth.particleCount(); ++i)
    {
        if ((w[i] > 0.0f && length(v[i]) > config.sleepSpeed) || i == grabIndex)
        {
            restless[blockGrid.blockOf(i % nx, i / nx)] = 1;
        }
    }
    // 2. sleepFrames 동안 연속으로 조용했던 블록만 잠든다.
    std::vector<unsigned char> seed(blockCount, 0);
    for (int b = 0; b < blockCount; ++b)
    {
        const bool flat = blockError[b] <= config.sleepError;
        const bool settled = std::fabs(blockError[b] - blockErrorPrev[b]) <= config.sleepErrorDrift;
        if (!flat && !settled)
        {
            restless[b] = 1;
        }
        calmFrames[b] = restless[b] ? 0 : std::min(calmFrames[b] + 1, 1 << 20);
        seed[b] = calmFrames[b] < config.sleepFrames ? 1 : 0;
    }
    firstFrame = false;
    activeBlocks = dilateBlocks(blockGrid, seed);
    std::vector<int> particles;
    for (int i = 0; i < cloth.particleCount(); ++i)
    {
        if (activeBlocks[blockGrid.blockOf(i % nx, i / nx)])
        {
            particles.push_back(i);
        }
    }
    stats.activeBlocks = static_cast<int>(std::count(activeBlocks.begin(), activeBlocks.end(), 1));
    if (stats.activeBlocks == 0)
    {
        stats.stepMs = ElapsedMs(stepStart);
        return stats;
    }
    // 대부분의 블록이 깨어 있으면 목록을 만드는 비용만 늘어나므로 전체 계산으로 전환한다.
    if (stats.activeBlocks > config.globalFallback * blockCount)
    {
        StepStats global = StepCpuGlobal(frameDt);
        std::fill(activeBlocks.begin(), activeBlocks.end(), static_cast<unsigned char>(1));
        UpdateBlockErrors(cloth, blockGrid, nullptr, activeBlocks, blockError, blockErrorPrev);
        global.activeBlocks = blockCount;
        global.totalBlocks = blockCount;
        global.globalFallback = true;
        global.stepMs = ElapsedMs(stepStart);
        return global;
    }
    // 활성 블록에 연결된 제약을 중복 없이 모은다.
    const std::vector<int> list = gatherConstraints(blockTable, activeBlocks);
    stats.solvedConstraints = static_cast<int>(list.size());

    const int substeps = std::max(1, config.substeps);
    const float dt = frameDt / static_cast<float>(substeps);
    const float damping = config.damping / static_cast<float>(substeps);

    double solveMs = 0.0;

    for (int k = 0; k < substeps; ++k)
    {
        // 활성 입자의 위치를 예측한다.
        for (size_t j = 0; j < particles.size(); j++)
        {
            int i = particles[j];
            if (i == grabIndex)
            {
                predicted[i] = grabPosition;
                continue;
            }
            if (w[i] == 0.0f)
            {
                predicted[i] = x[i];
                continue;
            }
            v[i] = (v[i] + config.gravity * dt) * (1.0f - damping);
            predicted[i] = x[i] + v[i] * dt;
        }

        // 활성 제약만 계산한다. 비활성 블록 쪽 끝점은 고정한다.
        const ClockTime solveStart = SimulationClock::now();
        for (int it = 0; it < config.maxIterations && !list.empty(); ++it)
        {
            for (size_t j = 0; j < list.size(); j++)
            {
                int c = list[j];
                const DistanceConstraint& ck = cloth.constraints[c];
                float wi = w[ck.i];
                float wj = w[ck.j];
                if (ck.i == grabIndex || !activeBlocks[blockGrid.blockOf(ck.i % nx, ck.i / nx)])
                {
                    wi = 0.0f;
                }
                if (ck.j == grabIndex || !activeBlocks[blockGrid.blockOf(ck.j % nx, ck.j / nx)])
                {
                    wj = 0.0f;
                }
                Vec3 dpi, dpj;
                if (!project_distance(predicted[ck.i], predicted[ck.j], wi, wj, ck.rest, ck.stiffness, dpi, dpj))
                {
                    continue;
                }
                if (wi > 0.0f)
                {
                    correction[ck.i] += dpi;
                    correctionCount[ck.i]++;
                }
                if (wj > 0.0f)
                {
                    correction[ck.j] += dpj;
                    correctionCount[ck.j]++;
                }
            }
            for (size_t j = 0; j < particles.size(); j++)
            {
                int i = particles[j];
                if (correctionCount[i])
                {
                    predicted[i] += correction[i] * (config.relaxation / static_cast<float>(correctionCount[i]));
                    correction[i] = Vec3{0, 0, 0};
                    correctionCount[i] = 0;
                }
            }
            stats.iterations++;
            if ((it + 1) % config.checkEvery == 0 || it + 1 == config.maxIterations)
            {
                float e = 0.0f;
                for (size_t j = 0; j < list.size(); j++)
                {
                    int c = list[j];
                    const DistanceConstraint& ck = cloth.constraints[c];
                    e = std::max(e, relative_error(predicted[ck.i], predicted[ck.j], ck.rest));
                }
                stats.maxError = e;
                if (e < config.tolerance)
                {
                    break;
                }
            }
        }
        solveMs += ElapsedMs(solveStart);

        const float inv = 1.0f / dt;
        for (size_t j = 0; j < particles.size(); j++)
        {
            int i = particles[j];
            v[i] = (w[i] != 0.0f && i != grabIndex) ? (predicted[i] - x[i]) * inv : Vec3{0, 0, 0};
            x[i] = predicted[i];
        }
    }
    // 3. 이번 프레임에 계산한 블록의 제약 오차를 갱신한다 (잠든 블록은 값이 그대로 유지된다).
    UpdateBlockErrors(cloth, blockGrid, &list, activeBlocks, blockError, blockErrorPrev);
    stats.solveMs = solveMs;
    stats.stepMs = ElapsedMs(stepStart);
    return stats;
}
}
