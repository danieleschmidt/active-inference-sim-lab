/*
 * Tests for free energy computation in active-inference-sim-lab.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

// Mock free energy class for testing
class FreeEnergy {
public:
    static double compute_accuracy_term(const std::vector<double>& observation,
                                      const std::vector<double>& prediction) {
        if (observation.size() != prediction.size()) {
            throw std::invalid_argument("Observation and prediction must have same size");
        }
        
        double sum_squared_error = 0.0;
        for (size_t i = 0; i < observation.size(); ++i) {
            double error = observation[i] - prediction[i];
            sum_squared_error += error * error;
        }
        
        return 0.5 * sum_squared_error;
    }
    
    static double compute_complexity_term(const std::vector<double>& posterior_mean,
                                        const std::vector<double>& prior_mean,
                                        double precision = 1.0) {
        if (posterior_mean.size() != prior_mean.size()) {
            throw std::invalid_argument("Posterior and prior must have same size");
        }
        
        double kl_divergence = 0.0;
        for (size_t i = 0; i < posterior_mean.size(); ++i) {
            double diff = posterior_mean[i] - prior_mean[i];
            kl_divergence += 0.5 * precision * diff * diff;
        }
        
        return kl_divergence;
    }
    
    static double compute_total_free_energy(const std::vector<double>& observation,
                                          const std::vector<double>& prediction,
                                          const std::vector<double>& posterior_mean,
                                          const std::vector<double>& prior_mean,
                                          double precision = 1.0) {
        double accuracy = compute_accuracy_term(observation, prediction);
        double complexity = compute_complexity_term(posterior_mean, prior_mean, precision);
        return accuracy + complexity;
    }
};

class FreeEnergyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test data
        observation = {1.0, 2.0, 3.0};
        prediction = {1.1, 1.9, 3.1};
        posterior_mean = {0.5, 1.0, 1.5};
        prior_mean = {0.0, 0.0, 0.0};
        precision = 1.0;
    }
    
    std::vector<double> observation;
    std::vector<double> prediction;
    std::vector<double> posterior_mean;
    std::vector<double> prior_mean;
    double precision;
};

TEST_F(FreeEnergyTest, AccuracyTermComputation) {
    double accuracy = FreeEnergy::compute_accuracy_term(observation, prediction);
    
    // Expected: 0.5 * (0.1^2 + 0.1^2 + 0.1^2) = 0.5 * 0.03 = 0.015
    EXPECT_NEAR(accuracy, 0.015, 1e-10);
}

TEST_F(FreeEnergyTest, ComplexityTermComputation) {
    double complexity = FreeEnergy::compute_complexity_term(posterior_mean, prior_mean, precision);
    
    // Expected: 0.5 * 1.0 * (0.5^2 + 1.0^2 + 1.5^2) = 0.5 * (0.25 + 1.0 + 2.25) = 1.75
    EXPECT_NEAR(complexity, 1.75, 1e-10);
}

TEST_F(FreeEnergyTest, TotalFreeEnergyComputation) {
    double total_fe = FreeEnergy::compute_total_free_energy(
        observation, prediction, posterior_mean, prior_mean, precision);
    
    // Expected: 0.015 + 1.75 = 1.765
    EXPECT_NEAR(total_fe, 1.765, 1e-10);
}

TEST_F(FreeEnergyTest, PerfectPredictionZeroAccuracy) {
    std::vector<double> perfect_prediction = observation;
    double accuracy = FreeEnergy::compute_accuracy_term(observation, perfect_prediction);
    
    EXPECT_NEAR(accuracy, 0.0, 1e-10);
}

TEST_F(FreeEnergyTest, PriorEqualsPosteriorZeroComplexity) {
    double complexity = FreeEnergy::compute_complexity_term(prior_mean, prior_mean, precision);
    
    EXPECT_NEAR(complexity, 0.0, 1e-10);
}

TEST_F(FreeEnergyTest, InvalidInputSizes) {
    std::vector<double> wrong_size = {1.0, 2.0};
    
    EXPECT_THROW(
        FreeEnergy::compute_accuracy_term(observation, wrong_size),
        std::invalid_argument
    );
    
    EXPECT_THROW(
        FreeEnergy::compute_complexity_term(posterior_mean, wrong_size),
        std::invalid_argument
    );
}

TEST_F(FreeEnergyTest, PrecisionScaling) {
    double low_precision = 0.1;
    double high_precision = 10.0;
    
    double complexity_low = FreeEnergy::compute_complexity_term(
        posterior_mean, prior_mean, low_precision);
    double complexity_high = FreeEnergy::compute_complexity_term(
        posterior_mean, prior_mean, high_precision);
    
    // Higher precision should lead to higher complexity penalty
    EXPECT_GT(complexity_high, complexity_low);
    EXPECT_NEAR(complexity_high, complexity_low * 100.0, 1e-10);
}

// Property-based tests
TEST(FreeEnergyProperties, NonNegativity) {
    std::vector<double> obs = {0.0, 1.0, -1.0};
    std::vector<double> pred = {0.1, 0.9, -1.1};
    std::vector<double> post = {0.0, 0.0, 0.0};
    std::vector<double> prior = {0.0, 0.0, 0.0};
    
    double accuracy = FreeEnergy::compute_accuracy_term(obs, pred);
    double complexity = FreeEnergy::compute_complexity_term(post, prior);
    double total = FreeEnergy::compute_total_free_energy(obs, pred, post, prior);
    
    EXPECT_GE(accuracy, 0.0);
    EXPECT_GE(complexity, 0.0);
    EXPECT_GE(total, 0.0);
}

TEST(FreeEnergyProperties, Monotonicity) {
    std::vector<double> obs = {1.0, 2.0};
    std::vector<double> pred1 = {1.0, 2.0};  // Perfect prediction
    std::vector<double> pred2 = {1.1, 2.1};  // Worse prediction
    
    double accuracy1 = FreeEnergy::compute_accuracy_term(obs, pred1);
    double accuracy2 = FreeEnergy::compute_accuracy_term(obs, pred2);
    
    // Worse prediction should have higher accuracy term
    EXPECT_GT(accuracy2, accuracy1);
}