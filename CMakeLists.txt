cmake_minimum_required(VERSION 3.15)
project(DataScience)


set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Options


# Inclusions
include_directories(include)


# Lib sources
set(LIB_SOURCES src/Data/Data.cpp
                src/Linalg/Linalg.cpp
                src/Linalg/LinalgNaive.cpp
)

# Main Library
add_library(linalg STATIC ${LIB_SOURCES})
target_compile_options(linalg PRIVATE -Wall -Wextra -O3 -march=native)


add_executable(Main src/main.cpp)
target_link_libraries(Main linalg)