/**
 * @file test_matrix_operations.cpp
 * @brief Tests for matrix operations and linear algebra
 */

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cmath>
#include <random>

class MatrixOperationsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up common test data
    generator_.seed(42);  // For reproducible tests
  }

  std::mt19937 generator_;
};

TEST_F(MatrixOperationsTest, BasicMatrixMultiplication) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 4);
  Eigen::MatrixXd B = Eigen::MatrixXd::Random(4, 2);

  Eigen::MatrixXd C = A * B;

  EXPECT_EQ(C.rows(), 3);
  EXPECT_EQ(C.cols(), 2);

  // Check that all elements are finite
  for (int i = 0; i < C.rows(); ++i) {
    for (int j = 0; j < C.cols(); ++j) {
      EXPECT_TRUE(std::isfinite(C(i, j)));
    }
  }
}

TEST_F(MatrixOperationsTest, MatrixInversion) {
  // Create a well-conditioned matrix
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(4, 4);
  A = A * A.transpose() + Eigen::MatrixXd::Identity(4, 4);  // Make positive definite

  Eigen::MatrixXd A_inv = A.inverse();

  // Check that A * A_inv ≈ I
  Eigen::MatrixXd product = A * A_inv;
  Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(4, 4);

  EXPECT_TRUE(product.isApprox(identity, 1e-10));
}

TEST_F(MatrixOperationsTest, EigenvalueDecomposition) {
  // Create symmetric matrix
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(5, 5);
  A = A + A.transpose();  // Make symmetric

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);

  EXPECT_EQ(solver.info(), Eigen::Success);

  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  Eigen::MatrixXd eigenvectors = solver.eigenvectors();

  // Check dimensions
  EXPECT_EQ(eigenvalues.size(), 5);
  EXPECT_EQ(eigenvectors.rows(), 5);
  EXPECT_EQ(eigenvectors.cols(), 5);

  // Check that eigenvalues are real (they should be for symmetric matrices)
  for (int i = 0; i < eigenvalues.size(); ++i) {
    EXPECT_TRUE(std::isfinite(eigenvalues(i)));
  }

  // Check that eigenvectors are orthonormal
  Eigen::MatrixXd should_be_identity = eigenvectors.transpose() * eigenvectors;
  EXPECT_TRUE(should_be_identity.isApprox(Eigen::MatrixXd::Identity(5, 5), 1e-10));
}

TEST_F(MatrixOperationsTest, LUDecomposition) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(4, 4);

  Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);

  // Check that we can solve linear systems
  Eigen::VectorXd b = Eigen::VectorXd::Random(4);
  Eigen::VectorXd x = lu.solve(b);

  // Check that A * x ≈ b
  Eigen::VectorXd residual = A * x - b;
  EXPECT_LT(residual.norm(), 1e-10);
}

TEST_F(MatrixOperationsTest, SVDDecomposition) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(6, 4);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::MatrixXd U = svd.matrixU();
  Eigen::VectorXd S = svd.singularValues();
  Eigen::MatrixXd V = svd.matrixV();

  // Check dimensions
  EXPECT_EQ(U.rows(), 6);
  EXPECT_EQ(U.cols(), 6);
  EXPECT_EQ(S.size(), 4);
  EXPECT_EQ(V.rows(), 4);
  EXPECT_EQ(V.cols(), 4);

  // Check that singular values are non-negative and sorted
  for (int i = 0; i < S.size(); ++i) {
    EXPECT_GE(S(i), 0.0);
    if (i > 0) {
      EXPECT_GE(S(i - 1), S(i));
    }
  }

  // Reconstruct matrix
  Eigen::MatrixXd S_matrix = Eigen::MatrixXd::Zero(6, 4);
  for (int i = 0; i < S.size(); ++i) {
    S_matrix(i, i) = S(i);
  }
  Eigen::MatrixXd A_reconstructed = U * S_matrix * V.transpose();

  EXPECT_TRUE(A.isApprox(A_reconstructed, 1e-10));
}

TEST_F(MatrixOperationsTest, CholeskyDecomposition) {
  // Create positive definite matrix
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(4, 4);
  A = A * A.transpose() + Eigen::MatrixXd::Identity(4, 4);

  Eigen::LLT<Eigen::MatrixXd> chol(A);

  EXPECT_EQ(chol.info(), Eigen::Success);

  Eigen::MatrixXd L = chol.matrixL();

  // Check that L * L^T = A
  Eigen::MatrixXd reconstructed = L * L.transpose();
  EXPECT_TRUE(A.isApprox(reconstructed, 1e-10));

  // Check that L is lower triangular
  for (int i = 0; i < L.rows(); ++i) {
    for (int j = i + 1; j < L.cols(); ++j) {
      EXPECT_NEAR(L(i, j), 0.0, 1e-10);
    }
  }
}

TEST_F(MatrixOperationsTest, NumericalStability) {
  // Test with very small numbers
  Eigen::MatrixXd A = 1e-10 * Eigen::MatrixXd::Random(3, 3);
  A = A * A.transpose() + 1e-12 * Eigen::MatrixXd::Identity(3, 3);

  Eigen::VectorXd b = 1e-10 * Eigen::VectorXd::Random(3);

  // Should still be able to solve the system
  Eigen::VectorXd x = A.ldlt().solve(b);

  for (int i = 0; i < x.size(); ++i) {
    EXPECT_TRUE(std::isfinite(x(i)));
  }

  // Check residual
  Eigen::VectorXd residual = A * x - b;
  EXPECT_LT(residual.norm() / b.norm(), 1e-6);  // Relative error
}

TEST_F(MatrixOperationsTest, LargeMatrixPerformance) {
  const int size = 100;
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(size, size);

  auto start = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXd result = A * A.transpose();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Should complete within reasonable time (1 second)
  EXPECT_LT(duration.count(), 1000);

  // Check result dimensions
  EXPECT_EQ(result.rows(), size);
  EXPECT_EQ(result.cols(), size);
}