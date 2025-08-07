#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <Eigen/Dense>

namespace active_inference {
namespace core {

/**
 * @brief Struct to hold components of free energy computation
 */
struct FreeEnergyComponents {
    double accuracy;
    double complexity;
    double total;
    
    FreeEnergyComponents(double acc, double comp) 
        : accuracy(acc), complexity(comp), total(acc + comp) {}
};

/**
 * @brief High-performance C++ implementation of Free Energy computation
 */
class FreeEnergy {
public:
    /**
     * @brief Constructor
     * @param complexity_weight Weight for complexity term
     * @param accuracy_weight Weight for accuracy term  
     * @param temperature Temperature parameter
     */
    FreeEnergy(double complexity_weight = 1.0, 
               double accuracy_weight = 1.0,
               double temperature = 1.0);
    
    /**
     * @brief Compute accuracy term (negative log-likelihood)
     * @param observations Observation vector
     * @param beliefs_mean Mean of belief distribution
     * @param beliefs_cov Covariance of belief distribution
     * @param likelihood_fn Likelihood function
     * @return Accuracy term
     */
    double compute_accuracy(const Eigen::VectorXd& observations,
                           const Eigen::VectorXd& beliefs_mean,
                           const Eigen::MatrixXd& beliefs_cov,
                           std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> likelihood_fn) const;
    
    /**
     * @brief Compute complexity term (KL divergence from prior)
     * @param beliefs_mean Current belief mean
     * @param beliefs_cov Current belief covariance
     * @param prior_mean Prior mean
     * @param prior_cov Prior covariance
     * @return Complexity term
     */
    double compute_complexity(const Eigen::VectorXd& beliefs_mean,
                             const Eigen::MatrixXd& beliefs_cov,
                             const Eigen::VectorXd& prior_mean,
                             const Eigen::MatrixXd& prior_cov) const;
    
    /**
     * @brief Compute total free energy and components
     * @param observations Current observations
     * @param beliefs_mean Belief mean
     * @param beliefs_cov Belief covariance
     * @param prior_mean Prior mean
     * @param prior_cov Prior covariance
     * @param likelihood_fn Observation likelihood function
     * @return Free energy components
     */
    FreeEnergyComponents compute_free_energy(const Eigen::VectorXd& observations,
                                           const Eigen::VectorXd& beliefs_mean,
                                           const Eigen::MatrixXd& beliefs_cov,
                                           const Eigen::VectorXd& prior_mean,
                                           const Eigen::MatrixXd& prior_cov,
                                           std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> likelihood_fn) const;

private:
    double complexity_weight_;
    double accuracy_weight_;
    double temperature_;
    
    /**
     * @brief Compute KL divergence between two multivariate Gaussians
     */
    double kl_divergence_gaussian(const Eigen::VectorXd& mu1, const Eigen::MatrixXd& sigma1,
                                 const Eigen::VectorXd& mu2, const Eigen::MatrixXd& sigma2) const;
    
    /**
     * @brief Sample from multivariate Gaussian distribution
     */
    std::vector<Eigen::VectorXd> sample_gaussian(const Eigen::VectorXd& mean,
                                                 const Eigen::MatrixXd& cov,
                                                 int n_samples) const;
};

}} // namespace active_inference::core