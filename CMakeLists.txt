cmake_minimum_required(VERSION 3.15)
project(DataScience)

set(CMAKE_CXX_STANDARD 20)


# COMPILER : MSVC vs MinGW-GCC
# ============================================
if(MSVC)
    add_compile_options(
        /W4         # /W4 to get warnings
        /O2         # /02 opti-max for balance size/speed
        /wd4127     # Constant conditionnal expression
        /wd4267     # size_t to int
        /wd4244     # double to int
        /wd5054)    # operator& between enums (Eigen) 
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native) # -Wall -Wextra to get warnings, -03 for performance, -march=native for opti 
endif()


# OPENMP
# ============================================
find_package(OpenMP REQUIRED)

if(NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP not found")
endif()


# THREAD
# ============================================
find_package(Threads REQUIRED)
if(NOT Threads_FOUND)
    message(FATAL_ERROR "Threads not found")
endif()

# AVX2
# ============================================
include(CheckCXXCompilerFlag)

if(MSVC)
    set(AVX2_FLAGS "/arch:AVX2")
    message(STATUS "✓ AVX2 (MSVC)")
else()
    CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)

    if(COMPILER_SUPPORTS_AVX2)
        set(AVX2_FLAGS "-mavx2")
        message(STATUS "✓ AVX2 (GCC)")
    else()
        set(AVX2_FLAGS "")
        message(WARNING "✗ AVX2")
    endif()
endif()


# INTEL MKL + TBB
# ============================================
option(USE_MKL "Use Intel MKL if available" ON)

set(MKL_FOUND FALSE)
set(TBB_FOUND FALSE)

if(USE_MKL AND MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    
    set(MKL_INTERFACE ilp64)
    set(MKL_THREADING tbb_thread) # Threading Building Blocks
    set(MKL_LINK static)

    find_package(MKL CONFIG QUIET)
    find_package(TBB CONFIG QUIET)

    if(MKL_FOUND AND TBB_FOUND)
        message(STATUS "✓ MKL & TBB enabled")
        message(STATUS "  Interface: ${MKL_INTERFACE}, Threading: ${MKL_THREADING}")
    else()
        message(WARNING "✗ MKL or/and TBB not found - continuing without")
        set(USE_MKL OFF)
    endif()
else()
    set(USE_MKL OFF)
    message(STATUS "Warning - MKL disabled by user or to avoid MinGW-GCC compiler (you may want to change to use MKL)")
endif()


# EIGEN
# ============================================
include(FetchContent)
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.1
)
FetchContent_MakeAvailable(Eigen)


# OPTIONS OF BUILD
# ============================================
option(BUILD_TEST "Build tests" ON)
option(BUILD_BENCHMARK "Build benchmark" ON)
option(BUILD_MAIN "Build main" ON)


# SOURCES
# ============================================
include_directories(include)

set(LIB_SOURCES 
    src/Data/Data.cpp
    src/Linalg/Linalg.cpp
    src/Linalg/LinalgNaive.cpp
    src/Linalg/LinalgAVX2.cpp
    src/Linalg/LinalgEigen.cpp
    src/Linalg/LinalgMKL.cpp
    src/Utils/TestSuite.cpp
    src/Utils/ThreadPool.cpp
)


# MAIN LIBRARY
# ============================================
add_library(linalg_full STATIC ${LIB_SOURCES})

target_compile_options(linalg_full PUBLIC ${AVX2_FLAGS})
target_link_libraries(linalg_full PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)

# Link Thread (MSVC do it automatically)
if(MINGW)
    target_link_libraries(linalg_full PUBLIC Threads::Threads)
endif()

# Link MKL + TBB
if(USE_MKL)
    target_compile_definitions(linalg_full PUBLIC USE_MKL MKL_ILP64 EIGEN_USE_MKL_ALL)
    target_link_libraries(linalg_full PUBLIC MKL::MKL TBB::tbb)
endif()


# TESTS
# ============================================
if(BUILD_TEST)
    enable_testing()
    add_executable(test_all 
        tests/tests_main.cpp
        tests/tests_data.cpp
        tests/tests_naive.cpp
        tests/tests_AVX2.cpp
        tests/tests_MKL.cpp
        tests/tests_Eigen.cpp
    )
    
    target_link_libraries(test_all linalg_full)
    add_test(NAME TestData COMMAND test_all data)
    add_test(NAME TestNaive COMMAND test_all naive)

    if(AVX2_FLAGS)
        add_test(NAME TestAVX2 COMMAND test_all avx2)
    endif()

    add_test(NAME TestEigen COMMAND test_all eigen)

    if(USE_MKL)
        add_test(NAME TestMKL COMMAND test_all mkl)
    endif()
endif()


# BENCHMARK
# ============================================
if(BUILD_BENCHMARK)
    set(BENCHMARK_ENABLE_TESTING OFF)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF)

    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main
    )
    FetchContent_MakeAvailable(benchmark)

    add_executable(my_bench 
        benchmark/bench_data.cpp
        #benchmark/bench_linalg.cpp
    )
    target_link_libraries(my_bench benchmark::benchmark_main linalg_full)    
endif()


# MAIN
# ============================================
if(BUILD_MAIN)
    add_executable(Main src/main.cpp)
    target_link_libraries(Main linalg_full)
endif()
