cmake_minimum_required(VERSION 3.18...3.25)
project(active_inference_sim_lab
    VERSION 0.1.0
    DESCRIPTION "Active Inference Simulation Laboratory - C++ Core"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -ffast-math
    )
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG -march=native)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4 /O2 /arch:AVX2)
    add_definitions(-D_USE_MATH_DEFINES)
endif()

# Find packages
find_package(Threads REQUIRED)

# Optional dependencies
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build C++ tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(USE_OPENMP "Enable OpenMP parallelization" ON)
option(USE_EIGEN "Use Eigen library for linear algebra" ON)
option(USE_BLAS "Use BLAS for linear algebra" OFF)

# OpenMP
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_compile_definitions(USE_OPENMP)
    endif()
endif()

# Eigen
if(USE_EIGEN)
    find_package(Eigen3 QUIET)
    if(NOT Eigen3_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(eigen)
    endif()
    add_compile_definitions(USE_EIGEN)
endif()

# BLAS
if(USE_BLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        add_compile_definitions(USE_BLAS)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Core library sources
set(CORE_SOURCES
    cpp/src/core/free_energy.cpp
)

# Create core static library
add_library(active_inference_core STATIC ${CORE_SOURCES})

# Link libraries
target_link_libraries(active_inference_core Threads::Threads)

if(OpenMP_CXX_FOUND)
    target_link_libraries(active_inference_core OpenMP::OpenMP_CXX)
endif()

if(USE_EIGEN AND TARGET Eigen3::Eigen)
    target_link_libraries(active_inference_core Eigen3::Eigen)
endif()

if(USE_BLAS AND BLAS_FOUND)
    target_link_libraries(active_inference_core ${BLAS_LIBRARIES})
endif()

# Set target properties
set_target_properties(active_inference_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "active_inference_core"
)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    # Find Python and pybind11
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    # Try to find pybind11
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.10.4
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    
    # Python module
    pybind11_add_module(_core cpp/src/bindings.cpp)
    target_link_libraries(_core PRIVATE active_inference_core)
    
    # Set module properties
    set_target_properties(_core PROPERTIES
        OUTPUT_NAME "_core"
        PREFIX ""
    )
    
    # Install the module
    install(TARGETS _core
        LIBRARY DESTINATION src/active_inference
        ARCHIVE DESTINATION src/active_inference
        RUNTIME DESTINATION src/active_inference
    )
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    
    # Find or fetch Google Test
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.12.1
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(googletest)
    endif()
    
    # Test sources
    set(TEST_SOURCES
        cpp/tests/test_free_energy.cpp
        cpp/tests/test_generative_model.cpp
        cpp/tests/test_belief_updater.cpp
        cpp/tests/test_active_planner.cpp
        cpp/tests/test_math_utils.cpp
    )
    
    # Create test executable
    add_executable(run_tests cpp/tests/main.cpp ${TEST_SOURCES})
    target_link_libraries(run_tests 
        active_inference_core
        gtest
        gtest_main
    )
    
    # Add tests
    add_test(NAME CoreTests COMMAND run_tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    # Find or fetch Google Benchmark
    find_package(benchmark QUIET)
    if(NOT benchmark_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googlebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.7.1
            GIT_SHALLOW TRUE
        )
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googlebenchmark)
    endif()
    
    # Benchmark sources
    set(BENCHMARK_SOURCES
        cpp/benchmarks/benchmark_free_energy.cpp
        cpp/benchmarks/benchmark_inference.cpp
        cpp/benchmarks/benchmark_planning.cpp
    )
    
    # Create benchmark executable
    add_executable(run_benchmarks ${BENCHMARK_SOURCES})
    target_link_libraries(run_benchmarks 
        active_inference_core
        benchmark::benchmark
        benchmark::benchmark_main
    )
endif()

# Installation
install(TARGETS active_inference_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY cpp/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Export configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ActiveInferenceConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ActiveInferenceConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ActiveInferenceConfig.cmake"
    INSTALL_DESTINATION lib/cmake/ActiveInference
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ActiveInferenceConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ActiveInferenceConfigVersion.cmake"
    DESTINATION lib/cmake/ActiveInference
)

# Summary
message(STATUS "")
message(STATUS "Active Inference Simulation Lab Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "  Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  OpenMP: ${USE_OPENMP}")
message(STATUS "  Eigen: ${USE_EIGEN}")
message(STATUS "  BLAS: ${USE_BLAS}")
message(STATUS "")