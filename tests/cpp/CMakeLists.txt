cmake_minimum_required(VERSION 3.16)

# Find GoogleTest
find_package(GTest REQUIRED)
include(GoogleTest)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Test sources
set(TEST_SOURCES
    test_main.cpp
    test_matrix_operations.cpp
    test_inference_engine.cpp
    test_planning_engine.cpp
)

# Create test executable
add_executable(test_runner ${TEST_SOURCES})

# Link libraries
target_link_libraries(test_runner
    PRIVATE
    GTest::gtest
    GTest::gtest_main
    Eigen3::Eigen
    ${CMAKE_THREAD_LIBS_INIT}
)

# Set C++ standard
target_compile_features(test_runner PRIVATE cxx_std_17)

# Add compiler flags
target_compile_options(test_runner PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -O2
)

# Discover tests
gtest_discover_tests(test_runner)

# Add custom test target
add_custom_target(run_tests
    COMMAND test_runner
    DEPENDS test_runner
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Add memory check target (if valgrind is available)
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    add_custom_target(test_memory
        COMMAND ${VALGRIND_EXECUTABLE} 
                --tool=memcheck 
                --leak-check=full 
                --show-leak-kinds=all 
                --track-origins=yes 
                --error-exitcode=1
                ./test_runner
        DEPENDS test_runner
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

# Add coverage target (if gcov is available)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    find_program(GCOV_EXECUTABLE gcov)
    find_program(LCOV_EXECUTABLE lcov)
    find_program(GENHTML_EXECUTABLE genhtml)
    
    if(GCOV_EXECUTABLE AND LCOV_EXECUTABLE AND GENHTML_EXECUTABLE)
        target_compile_options(test_runner PRIVATE --coverage)
        target_link_options(test_runner PRIVATE --coverage)
        
        add_custom_target(coverage
            COMMAND ${LCOV_EXECUTABLE} --directory . --capture --output-file coverage.info
            COMMAND ${LCOV_EXECUTABLE} --remove coverage.info '/usr/*' --output-file coverage.info
            COMMAND ${LCOV_EXECUTABLE} --list coverage.info
            COMMAND ${GENHTML_EXECUTABLE} coverage.info --output-directory coverage_html
            DEPENDS test_runner
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()
endif()