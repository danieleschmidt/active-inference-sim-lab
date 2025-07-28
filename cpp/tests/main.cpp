/*
 * Main test runner for C++ tests in active-inference-sim-lab.
 * 
 * This file sets up Google Test and runs all C++ unit tests.
 */

#include <gtest/gtest.h>
#include <iostream>

int main(int argc, char** argv) {
  std::cout << "Running Active Inference Sim Lab C++ Tests" << std::endl;
  
  ::testing::InitGoogleTest(&argc, argv);
  
  // Set up test environment
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  
  // Run all tests
  int result = RUN_ALL_TESTS();
  
  std::cout << "C++ tests completed with result: " << result << std::endl;
  
  return result;
}