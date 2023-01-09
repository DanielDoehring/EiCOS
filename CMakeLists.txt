cmake_minimum_required(VERSION 3.5)
project(eicos_LDBL)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fmt QUIET)

set(eicos_LDBL_INCLUDE
    include
)

set(eicos_LDBL_SOURCES
    src/eicos_LDBL.cpp
)

add_library(eicos_LDBL SHARED ${eicos_LDBL_SOURCES})
target_include_directories(eicos_LDBL PUBLIC ${eicos_LDBL_INCLUDE})

set(DEBUG_OPTIONS -Wall -Wextra -Wpedantic)
set(RELEASE_OPTIONS -Ofast -march=native)
target_compile_options(eicos_LDBL PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")
target_compile_options(eicos_LDBL PUBLIC "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>")

target_link_libraries(eicos_LDBL Eigen3::Eigen)

IF (${fmt_FOUND})
   MESSAGE(STATUS "Found fmt.")
   target_link_libraries(eicos_LDBL fmt::fmt)
ELSE (${fmt_FOUND})
   MESSAGE(STATUS "Could not locate fmt. Printing has been disabled.")
ENDIF (${fmt_FOUND})
