cmake_minimum_required(VERSION 3.16)

project(ActiveInferenceSimLab
    VERSION 0.1.0
    DESCRIPTION "Lightweight toolkit for building active inference agents based on the Free Energy Principle"
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

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Add compiler warnings
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Find required packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

# Optional packages
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_compile_definitions(HAVE_OPENMP)
endif()

# pybind11 for Python bindings
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    set(BUILD_PYTHON_BINDINGS ON CACHE BOOL "Build Python bindings")
else()
    set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Build Python bindings")
    message(STATUS "pybind11 not found, Python bindings will not be built")
endif()

# Testing
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "GoogleTest not found, tests will not be built")
        set(BUILD_TESTING OFF)
    endif()
endif()

# Documentation
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(BUILD_DOCS ON CACHE BOOL "Build documentation")
else()
    set(BUILD_DOCS OFF CACHE BOOL "Build documentation")
endif()

# Benchmarks
option(BUILD_BENCHMARKS "Build benchmarks" OFF)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source directories
add_subdirectory(src)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

# Tests
if(BUILD_TESTING)
    add_subdirectory(tests/cpp)
endif()

# Documentation
if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install configuration
include(GNUInstallDirs)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install cmake config
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ActiveInferenceSimLabConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ActiveInferenceSimLabConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ActiveInferenceSimLab
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ActiveInferenceSimLabConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ActiveInferenceSimLabConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ActiveInferenceSimLabConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ActiveInferenceSimLab
)

# CPack configuration
set(CPACK_PACKAGE_NAME "ActiveInferenceSimLab")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_VENDOR "Active Inference Research Team")
set(CPACK_PACKAGE_CONTACT "team@active-inference.org")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

include(CPack)

# Print configuration summary
message(STATUS "")
message(STATUS "Active Inference Simulation Lab Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Features:")
message(STATUS "  Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Testing: ${BUILD_TESTING}")
message(STATUS "  Documentation: ${BUILD_DOCS}")
message(STATUS "  Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "")