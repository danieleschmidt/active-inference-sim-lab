#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <chrono>

#include "core/free_energy.hpp"

namespace py = pybind11;
using namespace active_inference;

PYBIND11_MODULE(active_inference_cpp, m) {
    m.doc() = "High-performance C++ core for Active Inference Simulation Laboratory";
    
    // FreeEnergyComponents struct
    py::class_<core::FreeEnergyComponents>(m, "FreeEnergyComponents")
        .def(py::init<double, double>())
        .def_readwrite("accuracy", &core::FreeEnergyComponents::accuracy)
        .def_readwrite("complexity", &core::FreeEnergyComponents::complexity)
        .def_readwrite("total", &core::FreeEnergyComponents::total)
        .def("__repr__", [](const core::FreeEnergyComponents &fe) {
            return "FreeEnergyComponents(accuracy=" + std::to_string(fe.accuracy) +
                   ", complexity=" + std::to_string(fe.complexity) +
                   ", total=" + std::to_string(fe.total) + ")";
        });
    
    // FreeEnergy class
    py::class_<core::FreeEnergy>(m, "FreeEnergy")
        .def(py::init<double, double, double>(),
             py::arg("complexity_weight") = 1.0,
             py::arg("accuracy_weight") = 1.0, 
             py::arg("temperature") = 1.0)
        .def("compute_accuracy", &core::FreeEnergy::compute_accuracy,
             "Compute accuracy term (negative log-likelihood)",
             py::arg("observations"),
             py::arg("beliefs_mean"), 
             py::arg("beliefs_cov"),
             py::arg("likelihood_fn"))
        .def("compute_complexity", &core::FreeEnergy::compute_complexity,
             "Compute complexity term (KL divergence from prior)",
             py::arg("beliefs_mean"),
             py::arg("beliefs_cov"),
             py::arg("prior_mean"),
             py::arg("prior_cov"))
        .def("compute_free_energy", &core::FreeEnergy::compute_free_energy,
             "Compute total free energy and components",
             py::arg("observations"),
             py::arg("beliefs_mean"),
             py::arg("beliefs_cov"), 
             py::arg("prior_mean"),
             py::arg("prior_cov"),
             py::arg("likelihood_fn"));
    
    // Utility functions
    m.def("version", []() {
        return "0.1.0";
    }, "Get C++ core version");
    
    m.def("benchmark_free_energy", [](int n_dims, int n_iterations) {
        // Simple benchmark for performance testing
        auto fe = core::FreeEnergy();
        
        // Create test data
        Eigen::VectorXd obs = Eigen::VectorXd::Random(n_dims);
        Eigen::VectorXd beliefs_mean = Eigen::VectorXd::Random(n_dims);
        Eigen::MatrixXd beliefs_cov = Eigen::MatrixXd::Identity(n_dims, n_dims);
        Eigen::VectorXd prior_mean = Eigen::VectorXd::Zero(n_dims);
        Eigen::MatrixXd prior_cov = Eigen::MatrixXd::Identity(n_dims, n_dims);
        
        // Simple likelihood function
        auto likelihood_fn = [](const Eigen::VectorXd& state, const Eigen::VectorXd& observation) -> double {
            double error = (state - observation).norm();
            return std::exp(-0.5 * error * error);
        };
        
        // Run benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < n_iterations; ++i) {
            auto components = fe.compute_free_energy(obs, beliefs_mean, beliefs_cov, 
                                                   prior_mean, prior_cov, likelihood_fn);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return duration.count() / static_cast<double>(n_iterations); // microseconds per iteration
    }, "Benchmark free energy computation", py::arg("n_dims") = 10, py::arg("n_iterations") = 1000);
}