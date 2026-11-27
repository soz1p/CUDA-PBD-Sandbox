#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "pbd/solver.h"

using namespace pbd;


struct RunOptions
{
    std::vector<std::string> solvers{"cpu", "cpu-active"};
    int n = 96;
    int frames = 720;
    int warmup = 10;
    int cycle = 360;
    int pokeFrames = 30;
    int dumpEvery = 0;
    float dt = 1.0f / 60.0f, amplitude = 0.05f;
    SolverConfig cfg;
    std::string csv, dump;
    std::string pin = "frame";
};

static RunOptions ReadOptions(int argc, char** argv);
static void dumpFrame(const std::string& dir, const std::string& solver, int frame, const Cloth& c,
               const std::vector<unsigned char>& mask, int blockSize);

int main(int argc, char** argv)
{
    RunOptions options = ReadOptions(argc, argv);
    if (options.n < 2 || options.cycle < 1 || options.frames < 1 || options.warmup < 0 ||
        options.cfg.blockSize < 1)
    {
        std::fprintf(stderr, "Check --n, --cycle, --frames, --warmup and --block.\n");
        return 2;
    }

    ClothDesc desc;
    desc.nx = desc.ny = options.n;
    desc.spacing = 1.6f / static_cast<float>(options.n); // keep the cloth ~1.6 m wide at any resolution
    desc.pin = PinMode::Frame;
    if (options.pin == "top")
    {
        desc.pin = PinMode::TopEdge;
    }
    else if (options.pin == "corners")
    {
        desc.pin = PinMode::TopCorners;
    }
    Cloth start = Cloth::makeGrid(desc);

    // 모든 솔버가 같은 위치와 속도에서 시작하도록 천을 한 번만 안정화한다.
    {
        Solver settle;
        settle.Init(CPU_GLOBAL, options.cfg, start);
        for (int f = 0; f < options.warmup; ++f)
        {
            settle.Step(options.dt);
        }
        settle.GetCloth(start);
        for (int i = 0; i < start.particleCount(); i++)
        {
            start.velocity[i] = Vec3{0, 0, 0};
        }
    }

    // 매 주기마다 다음 위치를 누른다. 30프레임 뒤 손을 떼고 천이 가라앉도록 둔다.
    const int sites[4][2] = {{options.n / 2, options.n / 2},
                             {options.n / 4, options.n / 3},
                             {(options.n * 3) / 4, (options.n * 2) / 3},
                             {options.n / 3, (options.n * 3) / 4}};
    std::printf("cloth %dx%d, %d particles, %d constraints, %d frames, poke %d of every %d frames\n", options.n, options.n,
                start.particleCount(), start.constraintCount(), options.frames, options.pokeFrames, options.cycle);
    std::printf("%-11s %10s %10s %8s %10s %12s %12s %10s\n", "solver", "step ms", "solve ms", "iters", "active %",
                "constraints", "final err", "fallback %");

    std::ofstream csv;
    if (!options.csv.empty())
    {
        csv.open(options.csv);
        csv << "solver,frame,step_ms,solve_ms,iterations,max_error,active_blocks,total_blocks,solved_"
               "constraints,global_fallback\n";
    }

    for (size_t i = 0; i < options.solvers.size(); i++)
    {
        const std::string& kind = options.solvers[i];
#if defined(PBD_WITH_CUDA)
        if ((kind == "gpu" || kind == "active") && !cudaDeviceAvailable())
        {
            std::printf("%-11s skipped: no CUDA device\n", kind.c_str());
            continue;
        }
#endif
        SolverType type;
        if (kind == "gpu")
        {
            type = GPU_GLOBAL;
        }
        else if (kind == "active")
        {
            type = GPU_ACTIVE;
        }
        else if (kind == "cpu")
        {
            type = CPU_GLOBAL;
        }
        else if (kind == "cpu-active")
        {
            type = CPU_ACTIVE;
        }
        else
        {
            std::printf("unknown solver %s\n", kind.c_str());
            continue;
        }
        Solver solver;
        if (!solver.Init(type, options.cfg, start))
        {
            std::printf("%-11s not available in this build\n", kind.c_str());
            continue;
        }

        double stepMs = 0, solveMs = 0, iters = 0, activePct = 0, solved = 0, fallback = 0;
        for (int f = 0; f < options.frames; ++f)
        {
            const int phase = f % options.cycle, site = (f / options.cycle) % 4;
            if (phase < options.pokeFrames)
            {
                const int p = start.index(sites[site][0], sites[site][1]);
                const float u = static_cast<float>(phase) / static_cast<float>(options.pokeFrames);
                solver.SetGrab(p, start.position[p] +
                                      Vec3{0.0f, 0.0f, options.amplitude * std::sin(3.14159265f * u)});
            }
            else
            {
                solver.SetGrab(-1, Vec3{0, 0, 0});
            }
            const StepStats stats = solver.Step(options.dt);
            if (f > 0)
            { // 첫 프레임은 모든 블록을 깨우므로 평균에서 제외한다.
                stepMs += stats.stepMs;
                solveMs += stats.solveMs;
                iters += stats.iterations;
                solved += stats.solvedConstraints;
                activePct += stats.totalBlocks ? 100.0 * stats.activeBlocks / stats.totalBlocks : 100.0;
                fallback += stats.globalFallback ? 1.0 : 0.0;
            }
            if (csv.is_open())
            {
                csv << solver.GetName() << ',' << f << ',' << stats.stepMs << ',' << stats.solveMs << ','
                    << stats.iterations << ',' << stats.maxError << ',' << stats.activeBlocks << ','
                    << stats.totalBlocks << ',' << stats.solvedConstraints << ',' << (stats.globalFallback ? 1 : 0)
                    << '\n';
            }
            if (!options.dump.empty() && options.dumpEvery > 0 && f % options.dumpEvery == 0)
            {
                Cloth c = start;
                solver.GetCloth(c);
                dumpFrame(options.dump, kind, f, c, solver.GetActiveBlocks(), solver.GetBlockSize());
            }
        }
        Cloth end = start;
        solver.GetCloth(end);
        const double k = options.frames > 1 ? options.frames - 1 : 1;
        std::printf("%-11s %10.3f %10.3f %8.1f %10.1f %12.0f %12.2e %10.1f\n", solver.GetName(), stepMs / k,
                    solveMs / k, iters / k, activePct / k, solved / k, end.maxRelativeError(), 100.0 * fallback / k);
    }
    return 0;
}

static std::vector<std::string> split(const std::string& names)
{
    std::vector<std::string> out;
    std::stringstream ss(names);
    std::string name;
    while (std::getline(ss, name, ','))
    {
        if (!name.empty())
        {
            out.push_back(name);
        }
    }
    return out;
}

static const char* ReadValue(int argc, char** argv, int& index)
{
    if (index + 1 >= argc)
    {
        std::fprintf(stderr, "missing value for %s\n", argv[index]);
        std::exit(2);
    }
    index++;
    return argv[index];
}

static RunOptions ReadOptions(int argc, char** argv)
{
    RunOptions options;
    options.cfg.damping = 0.1f;
    options.cfg.gravity = Vec3{0.0f, 0.0f, 0.0f}; // 테두리를 고정하고 중력은 끈다.
#if defined(PBD_WITH_CUDA)
    if (cudaDeviceAvailable())
    {
        options.solvers = {"gpu", "active"};
    }
#endif
    for (int k = 1; k < argc; ++k)
    {
        const std::string option = argv[k];
        if (option == "--solvers")
        {
            options.solvers = split(ReadValue(argc, argv, k));
        }
        else if (option == "--n")
        {
            options.n = std::atoi(ReadValue(argc, argv, k));
        }
        else if (option == "--frames")
        {
            options.frames = std::atoi(ReadValue(argc, argv, k));
        }
        else if (option == "--warmup")
        {
            options.warmup = std::atoi(ReadValue(argc, argv, k));
        }
        else if (option == "--iters")
        {
            options.cfg.maxIterations = std::atoi(ReadValue(argc, argv, k));
        }
        else if (option == "--tol")
        {
            options.cfg.tolerance = std::strtof(ReadValue(argc, argv, k), nullptr);
        }
        else if (option == "--block")
        {
            options.cfg.blockSize = std::atoi(ReadValue(argc, argv, k));
        }
        else if (option == "--sleep")
        {
            options.cfg.sleepSpeed = std::strtof(ReadValue(argc, argv, k), nullptr);
        }
        else if (option == "--sleep-frames")
        {
            options.cfg.sleepFrames = std::atoi(ReadValue(argc, argv, k));
        }
        else if (option == "--sleep-error")
        {
            options.cfg.sleepError = std::strtof(ReadValue(argc, argv, k), nullptr);
        }
        else if (option == "--fallback")
        {
            options.cfg.globalFallback = std::strtof(ReadValue(argc, argv, k), nullptr);
        }
        else if (option == "--amp")
        {
            options.amplitude = std::strtof(ReadValue(argc, argv, k), nullptr);
        }
        else if (option == "--pin")
        {
            options.pin = ReadValue(argc, argv, k);
        }
        else if (option == "--cycle")
        {
            options.cycle = std::atoi(ReadValue(argc, argv, k));
        }
        else if (option == "--gravity")
        {
            options.cfg.gravity = Vec3{0.0f, -std::strtof(ReadValue(argc, argv, k), nullptr), 0.0f};
        }
        else if (option == "--poke")
        {
            options.pokeFrames = std::atoi(ReadValue(argc, argv, k));
        }
        else if (option == "--damping")
        {
            options.cfg.damping = std::strtof(ReadValue(argc, argv, k), nullptr);
        }
        else if (option == "--substeps")
        {
            options.cfg.substeps = std::atoi(ReadValue(argc, argv, k));
        }
        else if (option == "--csv")
        {
            options.csv = ReadValue(argc, argv, k);
        }
        else if (option == "--dump")
        {
            options.dump = ReadValue(argc, argv, k);
        }
        else if (option == "--dump-every")
        {
            options.dumpEvery = std::atoi(ReadValue(argc, argv, k));
        }
        else if (option == "-h" || option == "--help")
        {
            std::puts("pbd_bench [--solvers cpu,cpu-active,gpu,active] [--n 96] [--frames 720] [--warmup 10]\n"
                      "          [--substeps 8] [--iters 10] [--tol 1e-2] [--damping 0.1]\n"
                      "          [--block 8] [--sleep 0.002] [--sleep-frames 30] [--sleep-error 2e-3]\n"
                      "          [--fallback 0.7] [--pin frame|top|corners] [--gravity 0]\n"
                      "          [--cycle 360] [--poke 30] [--amp 0.05]\n"
                      "          [--csv out.csv] [--dump dir --dump-every 5]");
            std::exit(0);
        }
        else
        {
            std::fprintf(stderr, "unknown option %s\n", option.c_str());
            std::exit(2);
        }
    }
    return options;
}

// Text frame: "nx ny blockSize", then the active-block mask (0/1 string, may be empty),
// then one "x y z" line per particle.
static void dumpFrame(const std::string& dir, const std::string& solver, int frame, const Cloth& c,
               const std::vector<unsigned char>& mask, int blockSize)
{
    std::filesystem::create_directories(dir);
    char filename[32];
    std::snprintf(filename, sizeof(filename), "%s_%04d.xyz", solver.c_str(), frame);
    std::ofstream f(std::filesystem::path(dir) / filename);
    f << c.nx << ' ' << c.ny << ' ' << blockSize << '\n';
    for (size_t i = 0; i < mask.size(); i++)
    {
        f << (mask[i] ? '1' : '0');
    }
    f << '\n';
    for (int i = 0; i < c.particleCount(); i++)
    {
        const Vec3& position = c.position[i];
        f << position.x << ' ' << position.y << ' ' << position.z << '\n';
    }
}

