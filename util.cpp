#include "util.h"
#include <chrono>
#include <iomanip>

typedef decltype(std::chrono::steady_clock::now()) time_type;

struct TimeEstimate::Time
{
    time_type start;

    Time(time_type start) : start(std::move(start))
    {
    }
};

TimeEstimate::TimeEstimate() :
    start(make_unique<TimeEstimate::Time>(std::chrono::steady_clock::now()))
{
}

TimeEstimate::~TimeEstimate()
{
}

std::string TimeEstimate::Mark(int current, int total)
{
    auto now = std::chrono::steady_clock::now();
    auto current1 = current + 1;
    auto elapsed = now - start->start;
    auto ticks_per = elapsed / current1;
    auto frames_left = total - current1;
    auto time_left = ticks_per * frames_left;
    auto time_total = ticks_per * total;
    auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    auto elapsed_min = elapsed_sec / 60;
    elapsed_sec -= elapsed_min * 60;
    auto time_left_sec = std::chrono::duration_cast<std::chrono::seconds>(time_left).count();
    auto time_left_min = time_left_sec / 60;
    time_left_sec -= time_left_min * 60;
    auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(time_total).count();
    auto total_min = total_sec / 60;
    total_sec -= total_min * 60;
    std::ostringstream out;
    out
        << current1
        << "/"
        << total
        << " ("
        << (int)((100.0 * current1) / total)
        << "%), "
        << std::setfill('0')
        << elapsed_min
        << ":"
        << std::setw(2)
        << elapsed_sec
        << " elapsed, "
        << time_left_min
        << ":"
        << std::setw(2)
        << time_left_sec
        << " left, "
        << total_min
        << ":"
        << std::setw(2)
        << total_sec;
    return out.str();
}
