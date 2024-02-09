cmake_minimum_required(VERSION 3.5)
project(eicos_MP)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(EICOS_INCLUDE
    include
)

set(EICOS_SOURCES
    src/eicos_MP.cpp
)

add_library(eicos_MP SHARED ${EICOS_SOURCES})
target_include_directories(eicos_MP PUBLIC ${EICOS_INCLUDE})

set(DEBUG_OPTIONS -Wall -Wextra -Wpedantic)
set(RELEASE_OPTIONS -Ofast)
target_compile_options(eicos_MP PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")
target_compile_options(eicos_MP PUBLIC "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>")

target_link_libraries(eicos_MP Eigen3::Eigen)