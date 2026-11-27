#pragma once
#include <chrono>

namespace pbd
{
typedef std::chrono::steady_clock SimulationClock;
typedef SimulationClock::time_point ClockTime;

inline double ElapsedMs(ClockTime start)
{
    return std::chrono::duration<double, std::milli>(SimulationClock::now() - start).count();
}
}
