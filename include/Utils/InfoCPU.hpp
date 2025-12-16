#pragma once

#ifdef _MSC_VER // Microsoft
#include <intrin.h>
#else           // Linux/Mac
#include <cpuid.h>
#endif

// Getting CPU info to know if AVX and AVX2 is supported
inline bool hasAVX2() {
    int cpuInfo[4];
#ifdef _MSC_VER
    __cpuid(cpuInfo, 1);
    bool avxSupported = (cpuInfo[2] & (1 << 28)) != 0;
    __cpuid(cpuInfo, 7);
    bool avx2Supported = (cpuInfo[1] & (1 << 5)) != 0;
#else
    __cpuid(1, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
    bool avxSupported = (cpuInfo[2] & (1 << 28)) != 0;
    __cpuid_count(7, 0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
    bool avx2Supported = (cpuInfo[1] & (1 << 5)) != 0;
#endif
    return avxSupported && avx2Supported;
}