cmake_minimum_required(VERSION 3.15)
project(DataScience)


set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Options
option(BUILD_TEST "Build tests" ON)
option(BUILD_MAIN "Build main" ON)

# Inclusions
include_directories(include)


# Lib sources
set(LIB_SOURCES src/Data/Data.cpp
                src/Linalg/Linalg.cpp
                src/Linalg/LinalgNaive.cpp
                src/Utils/TestSuite.cpp
)


# Lib AVX2
set(LIB_AVX2_SOURCES src/Linalg/LinalgAVX2.cpp)


# Main Library
add_library(linalg STATIC ${LIB_SOURCES})
target_compile_options(linalg PRIVATE -Wall -Wextra -O3 -march=native)


# AVX2 Library
add_library(linalg_avx2 STATIC ${LIB_AVX2_SOURCES})
target_compile_options(linalg_avx2 PRIVATE -Wall -Wextra -O3 -mavx2 -mfma)


# Combine them 
add_library(linalg_full INTERFACE)
target_link_libraries(linalg_full INTERFACE linalg linalg_avx2)


if(BUILD_TEST)
    enable_testing()
    add_executable(test_all 
                tests/tests_main.cpp
                tests/tests_data.cpp
                tests/tests_naive.cpp
                tests/tests_AVX2.cpp)
                
    target_link_libraries(test_all linalg_full)
    add_test(NAME AllTests COMMAND test_all)
endif()


if(BUILD_MAIN)
    add_executable(Main src/main.cpp)
    target_link_libraries(Main linalg_full)
endif()