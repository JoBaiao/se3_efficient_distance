cmake_minimum_required(VERSION 3.16)
project(se3_efficient_distance VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(se3_efficient_distance
    src/se3_basic_math.cpp
    src/polynomial_curves.cpp
    src/analytical_distance.cpp
)

target_include_directories(se3_efficient_distance
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(se3_efficient_distance
    PUBLIC
        Eigen3::Eigen
)

target_compile_options(se3_efficient_distance PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

option(BUILD_EXAMPLES "Build minimal usage example" ON)
if(BUILD_EXAMPLES)
    add_executable(minimal_example
        examples/minimal_example.cpp
        examples/example_utils.cpp
    )
    target_link_libraries(minimal_example PRIVATE se3_efficient_distance)
    target_compile_definitions(minimal_example PRIVATE
        EXAMPLES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples"
    )
endif()