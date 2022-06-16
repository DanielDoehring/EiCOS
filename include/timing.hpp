#pragma once

#include <chrono>

long double tic()
{
    const std::chrono::duration<long double, std::milli> s = std::chrono::system_clock::now().time_since_epoch();

    return s.count();
}

long double toc(long double start)
{
    return tic() - start;
}
