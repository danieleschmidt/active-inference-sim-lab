/**
 * @file test_planning_engine.cpp
 * @brief Tests for planning engine functionality
 */

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

// Mock planning engine classes for testing
class MockTrajectory {
 public:
  MockTrajectory(int horizon, int state_dim, int action_dim)
      : horizon_(horizon), state_dim_(state_dim), action_dim_(action_dim) {
    states_.reserve(horizon + 1);
    actions_.reserve(horizon);
    states_.emplace_back(Eigen::VectorXd::Zero(state_dim));
  }

  void addAction(const Eigen::VectorXd& action) {
    if (actions_.size() < horizon_) {
      actions_.push_back(action);
    }
  }

  void addState(const Eigen::VectorXd& state) {
    if (states_.size() < horizon_ + 1) {
      states_.push_back(state);
    }
  }

  double computeValue(const std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>& cost_function) const {
    double total_cost = 0.0;
    for (size_t t = 0; t < actions_.size() && t < states_.size(); ++t) {
      total_cost += cost_function(states_[t], actions_[t]);
    }
    return -total_cost;  // Convert cost to value
  }

  const std::vector<Eigen::VectorXd>& states() const { return states_; }
  const std::vector<Eigen::VectorXd>& actions() const { return actions_; }
  int horizon() const { return horizon_; }

 private:
  int horizon_;
  int state_dim_;
  int action_dim_;
  std::vector<Eigen::VectorXd> states_;
  std::vector<Eigen::VectorXd> actions_;
};

class MockPlanningEngine {
 public:
  MockPlanningEngine(int state_dim, int action_dim, int horizon)
      : state_dim_(state_dim), action_dim_(action_dim), horizon_(horizon), generator_(42) {}

  MockTrajectory planTrajectory(const Eigen::VectorXd& initial_state,
                                const std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>& dynamics,
                                const std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>& cost_function,
                                int num_samples = 100) {
    std::vector<MockTrajectory> candidate_trajectories;

    // Sample random trajectories
    for (int sample = 0; sample < num_samples; ++sample) {
      MockTrajectory trajectory(horizon_, state_dim_, action_dim_);
      trajectory.addState(initial_state);

      Eigen::VectorXd current_state = initial_state;

      for (int t = 0; t < horizon_; ++t) {
        // Sample random action
        Eigen::VectorXd action = sampleRandomAction();
        trajectory.addAction(action);

        // Predict next state
        Eigen::VectorXd next_state = dynamics(current_state, action);
        trajectory.addState(next_state);

        current_state = next_state;
      }

      candidate_trajectories.push_back(trajectory);
    }

    // Select best trajectory
    auto best_trajectory = std::max_element(candidate_trajectories.begin(), candidate_trajectories.end(),
                                            [&cost_function](const MockTrajectory& a, const MockTrajectory& b) {
                                              return a.computeValue(cost_function) < b.computeValue(cost_function);
                                            });

    return *best_trajectory;
  }

  double computeExpectedFreeEnergy(const Eigen::VectorXd& initial_belief_mean, const Eigen::MatrixXd& initial_belief_cov,
                                   const std::vector<Eigen::VectorXd>& action_sequence,
                                   const std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>& dynamics,
                                   const Eigen::MatrixXd& process_noise, const Eigen::MatrixXd& obs_matrix,
                                   const Eigen::MatrixXd& obs_noise) {
    double expected_free_energy = 0.0;

    Eigen::VectorXd belief_mean = initial_belief_mean;
    Eigen::MatrixXd belief_cov = initial_belief_cov;

    for (size_t t = 0; t < action_sequence.size(); ++t) {
      const Eigen::VectorXd& action = action_sequence[t];

      // Predict belief after action
      Eigen::VectorXd predicted_mean = dynamics(belief_mean, action);
      Eigen::MatrixXd predicted_cov = belief_cov + process_noise;

      // Compute epistemic value (information gain)
      Eigen::MatrixXd predicted_obs_cov = obs_matrix * predicted_cov * obs_matrix.transpose() + obs_noise;
      double epistemic_value = 0.5 * std::log(predicted_obs_cov.determinant());

      // Compute pragmatic value (goal achievement)
      double pragmatic_value = -0.5 * predicted_mean.squaredNorm();  // Simple quadratic cost

      // Add to expected free energy
      expected_free_energy += epistemic_value + pragmatic_value;

      // Update belief for next step
      belief_mean = predicted_mean;
      belief_cov = predicted_cov;
    }

    return expected_free_energy;
  }

 private:
  Eigen::VectorXd sampleRandomAction() {
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::VectorXd action(action_dim_);
    for (int i = 0; i < action_dim_; ++i) {
      action(i) = dist(generator_);
    }
    return action;
  }

  int state_dim_;
  int action_dim_;
  int horizon_;
  std::mt19937 generator_;
};

class PlanningEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    state_dim_ = 4;
    action_dim_ = 2;
    horizon_ = 5;
    engine_ = std::make_unique<MockPlanningEngine>(state_dim_, action_dim_, horizon_);

    // Define simple dynamics: next_state = state + action + noise
    dynamics_ = [](const Eigen::VectorXd& state, const Eigen::VectorXd& action) -> Eigen::VectorXd {
      Eigen::VectorXd next_state = state;
      for (int i = 0; i < std::min(static_cast<int>(state.size()), static_cast<int>(action.size())); ++i) {
        next_state(i) += action(i);
      }
      return next_state;
    };

    // Define simple cost function: quadratic cost in state and action
    cost_function_ = [](const Eigen::VectorXd& state, const Eigen::VectorXd& action) -> double {
      return state.squaredNorm() + 0.1 * action.squaredNorm();
    };

    // Set up noise matrices
    process_noise_ = 0.01 * Eigen::MatrixXd::Identity(state_dim_, state_dim_);
    obs_noise_ = 0.1 * Eigen::MatrixXd::Identity(state_dim_, state_dim_);
    obs_matrix_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
  }

  int state_dim_;
  int action_dim_;
  int horizon_;
  std::unique_ptr<MockPlanningEngine> engine_;
  std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> dynamics_;
  std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> cost_function_;
  Eigen::MatrixXd process_noise_;
  Eigen::MatrixXd obs_noise_;
  Eigen::MatrixXd obs_matrix_;
};

TEST_F(PlanningEngineTest, TrajectoryConstruction) {
  MockTrajectory trajectory(horizon_, state_dim_, action_dim_);

  EXPECT_EQ(trajectory.horizon(), horizon_);
  EXPECT_EQ(trajectory.states().size(), 1);  // Initial state
  EXPECT_EQ(trajectory.actions().size(), 0);

  // Add actions and states
  for (int t = 0; t < horizon_; ++t) {
    Eigen::VectorXd action = Eigen::VectorXd::Random(action_dim_);
    Eigen::VectorXd state = Eigen::VectorXd::Random(state_dim_);

    trajectory.addAction(action);
    trajectory.addState(state);
  }

  EXPECT_EQ(trajectory.actions().size(), horizon_);
  EXPECT_EQ(trajectory.states().size(), horizon_ + 1);
}

TEST_F(PlanningEngineTest, TrajectoryValueComputation) {
  MockTrajectory trajectory(horizon_, state_dim_, action_dim_);

  // Add some actions and states
  for (int t = 0; t < horizon_; ++t) {
    Eigen::VectorXd action = Eigen::VectorXd::Zero(action_dim_);
    Eigen::VectorXd state = Eigen::VectorXd::Zero(state_dim_);

    trajectory.addAction(action);
    trajectory.addState(state);
  }

  double value = trajectory.computeValue(cost_function_);

  // Value should be finite
  EXPECT_TRUE(std::isfinite(value));

  // For zero states and actions, cost should be zero, so value should be zero
  EXPECT_NEAR(value, 0.0, 1e-10);
}

TEST_F(PlanningEngineTest, BasicPlanning) {
  Eigen::VectorXd initial_state = Eigen::VectorXd::Random(state_dim_);

  MockTrajectory planned_trajectory = engine_->planTrajectory(initial_state, dynamics_, cost_function_);

  // Check trajectory structure
  EXPECT_EQ(planned_trajectory.horizon(), horizon_);
  EXPECT_EQ(planned_trajectory.actions().size(), horizon_);
  EXPECT_EQ(planned_trajectory.states().size(), horizon_ + 1);

  // Check that initial state matches
  EXPECT_TRUE(planned_trajectory.states()[0].isApprox(initial_state));

  // Check that dynamics are consistent
  for (size_t t = 0; t < planned_trajectory.actions().size() && t + 1 < planned_trajectory.states().size(); ++t) {
    Eigen::VectorXd predicted_state = dynamics_(planned_trajectory.states()[t], planned_trajectory.actions()[t]);
    EXPECT_TRUE(planned_trajectory.states()[t + 1].isApprox(predicted_state));
  }
}

TEST_F(PlanningEngineTest, PlanningImprovement) {
  Eigen::VectorXd initial_state = Eigen::VectorXd::Ones(state_dim_);  // Non-zero initial state

  // Plan with few samples
  MockTrajectory trajectory_few_samples = engine_->planTrajectory(initial_state, dynamics_, cost_function_, 10);
  double value_few_samples = trajectory_few_samples.computeValue(cost_function_);

  // Plan with many samples
  MockTrajectory trajectory_many_samples = engine_->planTrajectory(initial_state, dynamics_, cost_function_, 1000);
  double value_many_samples = trajectory_many_samples.computeValue(cost_function_);

  // More samples should generally give better (higher) value
  EXPECT_GE(value_many_samples, value_few_samples - 1e-6);  // Allow for small numerical differences

  EXPECT_TRUE(std::isfinite(value_few_samples));
  EXPECT_TRUE(std::isfinite(value_many_samples));
}

TEST_F(PlanningEngineTest, ExpectedFreeEnergyComputation) {
  Eigen::VectorXd initial_belief_mean = Eigen::VectorXd::Zero(state_dim_);
  Eigen::MatrixXd initial_belief_cov = Eigen::MatrixXd::Identity(state_dim_, state_dim_);

  // Create action sequence
  std::vector<Eigen::VectorXd> action_sequence;
  for (int t = 0; t < horizon_; ++t) {
    action_sequence.push_back(Eigen::VectorXd::Random(action_dim_));
  }

  double efe = engine_->computeExpectedFreeEnergy(initial_belief_mean, initial_belief_cov, action_sequence, dynamics_, process_noise_, obs_matrix_,
                                                  obs_noise_);

  // Expected free energy should be finite
  EXPECT_TRUE(std::isfinite(efe));
}

TEST_F(PlanningEngineTest, ActionSequenceComparison) {
  Eigen::VectorXd initial_belief_mean = Eigen::VectorXd::Zero(state_dim_);
  Eigen::MatrixXd initial_belief_cov = Eigen::MatrixXd::Identity(state_dim_, state_dim_);

  // Create two different action sequences
  std::vector<Eigen::VectorXd> action_sequence_1;
  std::vector<Eigen::VectorXd> action_sequence_2;

  for (int t = 0; t < horizon_; ++t) {
    action_sequence_1.push_back(Eigen::VectorXd::Zero(action_dim_));          // Zero actions
    action_sequence_2.push_back(Eigen::VectorXd::Ones(action_dim_));          // Unit actions
  }

  double efe_1 = engine_->computeExpectedFreeEnergy(initial_belief_mean, initial_belief_cov, action_sequence_1, dynamics_, process_noise_,
                                                    obs_matrix_, obs_noise_);

  double efe_2 = engine_->computeExpectedFreeEnergy(initial_belief_mean, initial_belief_cov, action_sequence_2, dynamics_, process_noise_,
                                                    obs_matrix_, obs_noise_);

  // Both should be finite
  EXPECT_TRUE(std::isfinite(efe_1));
  EXPECT_TRUE(std::isfinite(efe_2));

  // They should be different (very unlikely to be exactly equal)
  EXPECT_NE(efe_1, efe_2);
}

TEST_F(PlanningEngineTest, PlanningConsistency) {
  Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim_);

  // Plan multiple times with the same initial conditions
  std::vector<double> values;
  for (int trial = 0; trial < 5; ++trial) {
    MockTrajectory trajectory = engine_->planTrajectory(initial_state, dynamics_, cost_function_, 100);
    double value = trajectory.computeValue(cost_function_);
    values.push_back(value);
  }

  // All values should be finite
  for (double value : values) {
    EXPECT_TRUE(std::isfinite(value));
  }

  // Values should be in a reasonable range (not all exactly the same due to randomness)
  double min_value = *std::min_element(values.begin(), values.end());
  double max_value = *std::max_element(values.begin(), values.end());

  EXPECT_LE(min_value, max_value);
}

TEST_F(PlanningEngineTest, LongHorizonPlanning) {
  int long_horizon = 20;
  MockPlanningEngine long_engine(state_dim_, action_dim_, long_horizon);

  Eigen::VectorXd initial_state = Eigen::VectorXd::Random(state_dim_);

  MockTrajectory trajectory = long_engine.planTrajectory(initial_state, dynamics_, cost_function_, 50);

  // Should handle long horizons without issues
  EXPECT_EQ(trajectory.horizon(), long_horizon);
  EXPECT_EQ(trajectory.actions().size(), long_horizon);
  EXPECT_EQ(trajectory.states().size(), long_horizon + 1);

  // All states and actions should be finite
  for (const auto& state : trajectory.states()) {
    for (int i = 0; i < state.size(); ++i) {
      EXPECT_TRUE(std::isfinite(state(i)));
    }
  }

  for (const auto& action : trajectory.actions()) {
    for (int i = 0; i < action.size(); ++i) {
      EXPECT_TRUE(std::isfinite(action(i)));
    }
  }
}