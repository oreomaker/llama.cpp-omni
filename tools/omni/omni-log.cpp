#include "omni-log.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>

#ifdef _WIN32
#    include <windows.h>
#endif

void print_with_timestamp(const char * format, ...) {
    auto now       = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms        = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm buf;
#ifdef _WIN32
    localtime_s(&buf, &in_time_t);
#else
    localtime_r(&in_time_t, &buf);
#endif

    std::cout << std::put_time(&buf, "%H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms.count() << " ";

    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}
