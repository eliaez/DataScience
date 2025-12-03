cmake_minimum_required(VERSION 4.2)
project(DataScience)

set(CMAKE_CXX_STANDARD 20)

add_executable(App 
    src/main.cpp
    src/Data/Data.cpp
    src/Linalg/Linalg.cpp)

target_include_directories(App PUBLIC ${CMAKE_SOURCE_DIR}/include)