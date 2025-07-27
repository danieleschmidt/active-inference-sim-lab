/**
 * @file test_inference_engine.cpp
 * @brief Tests for inference engine functionality
 */

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cmath>
#include <random>

// Mock inference engine classes for testing
class MockBeliefState {
 public:
  MockBeliefState(int dim) : mean_(Eigen::VectorXd::Zero(dim)), covariance_(Eigen::MatrixXd::Identity(dim, dim)) {}

  const Eigen::VectorXd& mean() const { return mean_; }
  const Eigen::MatrixXd& covariance() const { return covariance_; }

  void setMean(const Eigen::VectorXd& mean) { mean_ = mean; }
  void setCovariance(const Eigen::MatrixXd& cov) { covariance_ = cov; }

  double logLikelihood(const Eigen::VectorXd& observation, const Eigen::MatrixXd& obs_noise) const {
    Eigen::VectorXd diff = observation - mean_;
    return -0.5 * (diff.transpose() * obs_noise.inverse() * diff).value();
  }

  void update(const Eigen::VectorXd& observation, const Eigen::MatrixXd& obs_matrix,
              const Eigen::MatrixXd& obs_noise) {
    // Kalman filter update
    Eigen::MatrixXd S = obs_matrix * covariance_ * obs_matrix.transpose() + obs_noise;
    Eigen::MatrixXd K = covariance_ * obs_matrix.transpose() * S.inverse();

    Eigen::VectorXd innovation = observation - obs_matrix * mean_;
    mean_ = mean_ + K * innovation;
    covariance_ = covariance_ - K * obs_matrix * covariance_;
  }

 private:
  Eigen::VectorXd mean_;
  Eigen::MatrixXd covariance_;
};

class MockInferenceEngine {
 public:
  MockInferenceEngine(int state_dim, int obs_dim) : state_dim_(state_dim), obs_dim_(obs_dim) {}

  MockBeliefState performInference(const Eigen::VectorXd& observation, const MockBeliefState& prior,
                                   const Eigen::MatrixXd& transition_matrix, const Eigen::MatrixXd& obs_matrix,
                                   const Eigen::MatrixXd& process_noise, const Eigen::MatrixXd& obs_noise) {
    // Predict step
    Eigen::VectorXd predicted_mean = transition_matrix * prior.mean();
    Eigen::MatrixXd predicted_cov = transition_matrix * prior.covariance() * transition_matrix.transpose() + process_noise;

    // Update step
    MockBeliefState predicted_belief(state_dim_);
    predicted_belief.setMean(predicted_mean);
    predicted_belief.setCovariance(predicted_cov);

    MockBeliefState posterior_belief = predicted_belief;
    posterior_belief.update(observation, obs_matrix, obs_noise);

    return posterior_belief;
  }

  double computeFreeEnergy(const MockBeliefState& belief, const Eigen::VectorXd& observation,
                           const Eigen::MatrixXd& obs_matrix, const Eigen::MatrixXd& obs_noise,
                           const MockBeliefState& prior) {
    // Accuracy term (negative log likelihood)
    double accuracy = -belief.logLikelihood(observation, obs_noise);

    // Complexity term (KL divergence from prior)
    Eigen::VectorXd mean_diff = belief.mean() - prior.mean();
    Eigen::MatrixXd cov_inv_prior = prior.covariance().inverse();

    double complexity = 0.5 * (cov_inv_prior * belief.covariance()).trace() + 0.5 * (mean_diff.transpose() * cov_inv_prior * mean_diff).value() -
                        0.5 * state_dim_ + 0.5 * std::log(belief.covariance().determinant() / prior.covariance().determinant());

    return accuracy + complexity;
  }

 private:
  int state_dim_;
  int obs_dim_;
};

class InferenceEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    state_dim_ = 4;
    obs_dim_ = 2;
    engine_ = std::make_unique<MockInferenceEngine>(state_dim_, obs_dim_);

    // Set up matrices
    transition_matrix_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
    transition_matrix_(0, 1) = 0.1;  // Simple dynamics

    obs_matrix_ = Eigen::MatrixXd::Zero(obs_dim_, state_dim_);
    obs_matrix_(0, 0) = 1.0;
    obs_matrix_(1, 2) = 1.0;

    process_noise_ = 0.01 * Eigen::MatrixXd::Identity(state_dim_, state_dim_);
    obs_noise_ = 0.1 * Eigen::MatrixXd::Identity(obs_dim_, obs_dim_);
  }

  int state_dim_;
  int obs_dim_;
  std::unique_ptr<MockInferenceEngine> engine_;
  Eigen::MatrixXd transition_matrix_;
  Eigen::MatrixXd obs_matrix_;
  Eigen::MatrixXd process_noise_;
  Eigen::MatrixXd obs_noise_;
};

TEST_F(InferenceEngineTest, BeliefStateConstruction) {
  MockBeliefState belief(state_dim_);

  EXPECT_EQ(belief.mean().size(), state_dim_);
  EXPECT_EQ(belief.covariance().rows(), state_dim_);
  EXPECT_EQ(belief.covariance().cols(), state_dim_);

  // Should be initialized to zero mean and identity covariance
  EXPECT_TRUE(belief.mean().isZero());
  EXPECT_TRUE(belief.covariance().isApprox(Eigen::MatrixXd::Identity(state_dim_, state_dim_)));
}

TEST_F(InferenceEngineTest, BeliefStateUpdate) {
  MockBeliefState belief(state_dim_);

  // Create observation
  Eigen::VectorXd observation(obs_dim_);
  observation << 1.0, -0.5;

  // Update belief
  belief.update(observation, obs_matrix_, obs_noise_);

  // Mean should have changed
  EXPECT_FALSE(belief.mean().isZero());

  // Covariance should have decreased (uncertainty reduction)
  Eigen::MatrixXd updated_cov = belief.covariance();
  for (int i = 0; i < state_dim_; ++i) {
    EXPECT_LE(updated_cov(i, i), 1.0);  // Diagonal elements should be <= 1
  }

  // All values should be finite
  for (int i = 0; i < state_dim_; ++i) {
    EXPECT_TRUE(std::isfinite(belief.mean()(i)));
    for (int j = 0; j < state_dim_; ++j) {
      EXPECT_TRUE(std::isfinite(belief.covariance()(i, j)));
    }
  }
}

TEST_F(InferenceEngineTest, InferenceStep) {
  MockBeliefState prior(state_dim_);
  prior.setMean(Eigen::VectorXd::Random(state_dim_));

  Eigen::VectorXd observation(obs_dim_);
  observation << 0.5, -1.0;

  MockBeliefState posterior = engine_->performInference(observation, prior, transition_matrix_, obs_matrix_, process_noise_, obs_noise_);

  // Posterior should be different from prior
  EXPECT_FALSE(posterior.mean().isApprox(prior.mean()));
  EXPECT_FALSE(posterior.covariance().isApprox(prior.covariance()));

  // All values should be finite
  for (int i = 0; i < state_dim_; ++i) {
    EXPECT_TRUE(std::isfinite(posterior.mean()(i)));
    for (int j = 0; j < state_dim_; ++j) {
      EXPECT_TRUE(std::isfinite(posterior.covariance()(i, j)));
    }
  }
}

TEST_F(InferenceEngineTest, FreeEnergyComputation) {
  MockBeliefState belief(state_dim_);
  belief.setMean(Eigen::VectorXd::Random(state_dim_));

  MockBeliefState prior(state_dim_);

  Eigen::VectorXd observation(obs_dim_);
  observation << 0.0, 0.0;

  double free_energy = engine_->computeFreeEnergy(belief, observation, obs_matrix_, obs_noise_, prior);

  // Free energy should be finite
  EXPECT_TRUE(std::isfinite(free_energy));

  // Free energy should be positive (typically)
  EXPECT_GE(free_energy, 0.0);
}

TEST_F(InferenceEngineTest, SequentialInference) {
  MockBeliefState belief(state_dim_);

  // Sequence of observations
  std::vector<Eigen::VectorXd> observations;
  for (int t = 0; t < 10; ++t) {
    Eigen::VectorXd obs(obs_dim_);
    obs << std::sin(0.1 * t), std::cos(0.1 * t);
    observations.push_back(obs);
  }

  // Track belief evolution
  std::vector<MockBeliefState> belief_sequence;
  belief_sequence.push_back(belief);

  for (const auto& obs : observations) {
    belief = engine_->performInference(obs, belief, transition_matrix_, obs_matrix_, process_noise_, obs_noise_);
    belief_sequence.push_back(belief);

    // Check that belief remains valid
    for (int i = 0; i < state_dim_; ++i) {
      EXPECT_TRUE(std::isfinite(belief.mean()(i)));
      for (int j = 0; j < state_dim_; ++j) {
        EXPECT_TRUE(std::isfinite(belief.covariance()(i, j)));
      }
    }
  }

  EXPECT_EQ(belief_sequence.size(), observations.size() + 1);
}

TEST_F(InferenceEngineTest, UncertaintyReduction) {
  MockBeliefState belief(state_dim_);

  // Initial uncertainty
  double initial_uncertainty = belief.covariance().trace();

  // Multiple observations of the same state
  Eigen::VectorXd consistent_obs(obs_dim_);
  consistent_obs << 1.0, 1.0;

  for (int i = 0; i < 5; ++i) {
    belief.update(consistent_obs, obs_matrix_, obs_noise_);
  }

  // Final uncertainty should be lower
  double final_uncertainty = belief.covariance().trace();
  EXPECT_LT(final_uncertainty, initial_uncertainty);
}

TEST_F(InferenceEngineTest, NumericalStability) {
  MockBeliefState belief(state_dim_);

  // Very noisy observations
  Eigen::MatrixXd high_noise = 100.0 * Eigen::MatrixXd::Identity(obs_dim_, obs_dim_);

  Eigen::VectorXd noisy_obs(obs_dim_);
  noisy_obs << 1000.0, -1000.0;

  belief.update(noisy_obs, obs_matrix_, high_noise);

  // Should remain numerically stable
  for (int i = 0; i < state_dim_; ++i) {
    EXPECT_TRUE(std::isfinite(belief.mean()(i)));
    for (int j = 0; j < state_dim_; ++j) {
      EXPECT_TRUE(std::isfinite(belief.covariance()(i, j)));
    }
  }

  // Covariance should remain positive definite
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(belief.covariance());
  for (int i = 0; i < state_dim_; ++i) {
    EXPECT_GT(eigen_solver.eigenvalues()(i), -1e-10);  // Allow for small numerical errors
  }
}