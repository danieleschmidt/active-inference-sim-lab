#include "free_energy.hpp"
#include <random>
#include <cmath>
#include <stdexcept>

namespace active_inference {
namespace core {

FreeEnergy::FreeEnergy(double complexity_weight, double accuracy_weight, double temperature)
    : complexity_weight_(complexity_weight), accuracy_weight_(accuracy_weight), temperature_(temperature) {
    if (complexity_weight_ < 0 || accuracy_weight_ < 0 || temperature_ <= 0) {
        throw std::invalid_argument("Weights must be non-negative and temperature must be positive");
    }
}

double FreeEnergy::compute_accuracy(const Eigen::VectorXd& observations,
                                   const Eigen::VectorXd& beliefs_mean,
                                   const Eigen::MatrixXd& beliefs_cov,
                                   std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> likelihood_fn) const {
    
    if (beliefs_mean.size() == 0) {
        return std::numeric_limits<double>::infinity();
    }
    
    // Monte Carlo estimation of expected log-likelihood
    const int n_samples = 100;
    auto samples = sample_gaussian(beliefs_mean, beliefs_cov, n_samples);
    
    double total_log_likelihood = 0.0;
    
    for (const auto& sample : samples) {
        double likelihood = likelihood_fn(sample, observations);
        if (likelihood <= 0) {
            likelihood = 1e-8; // Numerical stability
        }
        total_log_likelihood += std::log(likelihood);
    }
    
    double expected_log_likelihood = total_log_likelihood / n_samples;
    
    // Return negative log-likelihood (minimize = maximize likelihood)
    return -expected_log_likelihood * accuracy_weight_;
}

double FreeEnergy::compute_complexity(const Eigen::VectorXd& beliefs_mean,
                                     const Eigen::MatrixXd& beliefs_cov,
                                     const Eigen::VectorXd& prior_mean,
                                     const Eigen::MatrixXd& prior_cov) const {
    
    return kl_divergence_gaussian(beliefs_mean, beliefs_cov, prior_mean, prior_cov) * complexity_weight_;
}

FreeEnergyComponents FreeEnergy::compute_free_energy(const Eigen::VectorXd& observations,
                                                   const Eigen::VectorXd& beliefs_mean,
                                                   const Eigen::MatrixXd& beliefs_cov,
                                                   const Eigen::VectorXd& prior_mean,
                                                   const Eigen::MatrixXd& prior_cov,
                                                   std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> likelihood_fn) const {
    
    double accuracy = compute_accuracy(observations, beliefs_mean, beliefs_cov, likelihood_fn);
    double complexity = compute_complexity(beliefs_mean, beliefs_cov, prior_mean, prior_cov);
    
    return FreeEnergyComponents(accuracy, complexity);
}

double FreeEnergy::kl_divergence_gaussian(const Eigen::VectorXd& mu1, const Eigen::MatrixXd& sigma1,
                                         const Eigen::VectorXd& mu2, const Eigen::MatrixXd& sigma2) const {
    
    if (mu1.size() != mu2.size() || sigma1.rows() != sigma2.rows() || sigma1.cols() != sigma2.cols()) {
        throw std::invalid_argument("Dimension mismatch in KL divergence computation");
    }
    
    int k = mu1.size();
    
    // Ensure positive definite matrices
    Eigen::MatrixXd sigma2_reg = sigma2 + 1e-6 * Eigen::MatrixXd::Identity(k, k);
    Eigen::MatrixXd sigma1_reg = sigma1 + 1e-6 * Eigen::MatrixXd::Identity(k, k);
    
    // Compute inverse of sigma2
    Eigen::MatrixXd sigma2_inv = sigma2_reg.inverse();
    
    // Mean difference
    Eigen::VectorXd mu_diff = mu1 - mu2;
    
    // KL divergence components
    double trace_term = (sigma2_inv * sigma1_reg).trace();
    double quadratic_term = mu_diff.transpose() * sigma2_inv * mu_diff;
    double log_det_term = std::log(sigma2_reg.determinant()) - std::log(sigma1_reg.determinant());
    
    double kl = 0.5 * (trace_term + quadratic_term - k + log_det_term);
    
    return std::max(0.0, kl); // Ensure non-negative
}

std::vector<Eigen::VectorXd> FreeEnergy::sample_gaussian(const Eigen::VectorXd& mean,
                                                        const Eigen::MatrixXd& cov,
                                                        int n_samples) const {
    
    std::vector<Eigen::VectorXd> samples;
    samples.reserve(n_samples);
    
    // Cholesky decomposition for sampling
    Eigen::LLT<Eigen::MatrixXd> llt(cov);
    if (llt.info() != Eigen::Success) {
        // If Cholesky fails, add regularization
        Eigen::MatrixXd cov_reg = cov + 1e-6 * Eigen::MatrixXd::Identity(cov.rows(), cov.cols());
        llt.compute(cov_reg);
    }
    
    Eigen::MatrixXd L = llt.matrixL();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < n_samples; ++i) {
        Eigen::VectorXd z(mean.size());
        for (int j = 0; j < mean.size(); ++j) {
            z[j] = dist(gen);
        }
        
        Eigen::VectorXd sample = mean + L * z;
        samples.push_back(sample);
    }
    
    return samples;
}

}} // namespace active_inference::core