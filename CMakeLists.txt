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

# Main Library
add_library(linalg STATIC ${LIB_SOURCES})
target_compile_options(linalg PRIVATE -Wall -Wextra -O3 -march=native)

if(BUILD_TEST)
    enable_testing()
    add_executable(test_all 
                tests/tests_main.cpp
                tests/tests_data.cpp)
                
    target_link_libraries(test_all linalg)
    add_test(NAME AllTests COMMAND test_all)
endif()

if(BUILD_MAIN)
    add_executable(Main src/main.cpp)
    target_link_libraries(Main linalg)
endif()