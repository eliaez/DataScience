cmake_minimum_required(VERSION 3.15)
project(DataScience)


set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect AVX2
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX2)
    set(AVX2_FLAGS "-mavx2")
    message(STATUS "✓ AVX2")
else()
    set(AVX2_FLAGS "")
    message(WARNING "✗ AVX2")
endif()

# Add Eigen
include(FetchContent)
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.1
)
FetchContent_MakeAvailable(Eigen)

# Options
option(BUILD_TEST "Build tests" OFF)
option(BUILD_BENCHMARK "Build benchmark" ON)
option(BUILD_MAIN "Build main" OFF)


# Inclusions
include_directories(include)


# Lib sources
set(LIB_SOURCES src/Data/Data.cpp
                src/Linalg/Linalg.cpp
                src/Linalg/LinalgNaive.cpp
                src/Linalg/LinalgAVX2.cpp
                src/Utils/TestSuite.cpp
)


# Main Library
add_library(linalg_full STATIC ${LIB_SOURCES})
target_compile_options(linalg_full PRIVATE -Wall -Wextra -O3 -march=native ${AVX2_FLAGS})
target_link_libraries(linalg_full PUBLIC Eigen3::Eigen)

if(BUILD_TEST)
    enable_testing()
    add_executable(test_all 
                tests/tests_main.cpp
                tests/tests_data.cpp
                tests/tests_naive.cpp
                tests/tests_AVX2.cpp)
    
    target_compile_options(test_all PRIVATE ${AVX2_FLAGS})
    target_link_libraries(test_all linalg_full)
    add_test(NAME AllTests COMMAND test_all)
endif()


# Get and build benchmark
if(BUILD_BENCHMARK)
    include(FetchContent)

    set(BENCHMARK_ENABLE_TESTING OFF)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF)

    FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG main
    )
    FetchContent_MakeAvailable(benchmark)

    add_executable(my_bench benchmark/bench_data.cpp
                            benchmark/bench_linalg.cpp)
                            
    target_link_libraries(my_bench benchmark::benchmark_main linalg_full)    
endif()


if(BUILD_MAIN)
    add_executable(Main src/main.cpp)
    target_compile_options(Main PRIVATE ${AVX2_FLAGS})
    target_link_libraries(Main linalg_full)
endif()