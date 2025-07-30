# Performance Benchmarking Automation

## Overview

This document provides comprehensive performance benchmarking automation for the Active Inference Sim Lab, including micro-benchmarks, integration benchmarks, and continuous performance monitoring.

## Benchmark Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Micro         │    │   Integration   │    │   System        │
│   Benchmarks    │───▶│   Benchmarks    │───▶│   Benchmarks    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Results       │    │   Tracking      │    │   Alerting      │
│   Collection    │───▶│   & Analysis    │───▶│   & Reporting   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 1. C++ Benchmark Framework

Create `cpp/benchmarks/benchmark_framework.hpp`:

```cpp
#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <functional>
#include <map>
#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

namespace active_inference {
namespace benchmarks {

// Benchmark result structure
struct BenchmarkResult {
    std::string name;
    double mean_time_ms;
    double std_dev_ms;
    double min_time_ms;
    double max_time_ms;
    size_t iterations;
    std::map<std::string, double> custom_metrics;
};

// Benchmark configuration
struct BenchmarkConfig {
    size_t warmup_iterations = 10;
    size_t benchmark_iterations = 100;
    double time_limit_seconds = 60.0;
    bool auto_adjust_iterations = true;
    std::string output_format = "json"; // json, csv, human
};

class BenchmarkTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool running_ = false;

public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        running_ = true;
    }
    
    void stop() {
        end_time_ = std::chrono::high_resolution_clock::now();
        running_ = false;
    }
    
    double elapsed_milliseconds() const {
        if (running_) {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(now - start_time_).count();
        }
        return std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
    }
};

class BenchmarkSuite {
private:
    std::vector<std::pair<std::string, std::function<void()>>> benchmarks_;
    BenchmarkConfig config_;
    std::vector<BenchmarkResult> results_;

public:
    explicit BenchmarkSuite(const BenchmarkConfig& config = BenchmarkConfig{})
        : config_(config) {}
    
    // Register a benchmark function
    void add_benchmark(const std::string& name, std::function<void()> benchmark_func) {
        benchmarks_.emplace_back(name, std::move(benchmark_func));
    }
    
    // Run all benchmarks
    void run_all() {
        results_.clear();
        results_.reserve(benchmarks_.size());
        
        std::cout << "Running " << benchmarks_.size() << " benchmarks..." << std::endl;
        
        for (const auto& [name, func] : benchmarks_) {
            std::cout << "Benchmarking: " << name << "..." << std::flush;
            auto result = run_single_benchmark(name, func);
            results_.push_back(result);
            std::cout << " " << std::fixed << std::setprecision(3) 
                      << result.mean_time_ms << " ms" << std::endl;
        }
    }
    
    // Get benchmark results
    const std::vector<BenchmarkResult>& get_results() const {
        return results_;
    }
    
    // Export results in specified format
    void export_results(const std::string& filename) const {
        if (config_.output_format == "json") {
            export_json(filename);
        } else if (config_.output_format == "csv") {
            export_csv(filename);
        } else {
            export_human_readable(filename);
        }
    }

private:
    BenchmarkResult run_single_benchmark(const std::string& name, 
                                        const std::function<void()>& func) {
        std::vector<double> times;
        times.reserve(config_.benchmark_iterations);
        
        // Warmup phase
        for (size_t i = 0; i < config_.warmup_iterations; ++i) {
            func();
        }
        
        BenchmarkTimer timer;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Benchmark phase
        for (size_t i = 0; i < config_.benchmark_iterations; ++i) {
            timer.start();
            func();
            timer.stop();
            
            times.push_back(timer.elapsed_milliseconds());
            
            // Check time limit
            if (config_.auto_adjust_iterations) {
                auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
                auto elapsed_seconds = std::chrono::duration<double>(elapsed).count();
                if (elapsed_seconds > config_.time_limit_seconds) {
                    break;
                }
            }
        }
        
        // Calculate statistics
        double mean = 0.0;
        double min_time = times[0];
        double max_time = times[0];
        
        for (double time : times) {
            mean += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        mean /= times.size();
        
        double variance = 0.0;
        for (double time : times) {
            variance += (time - mean) * (time - mean);
        }
        variance /= times.size();
        double std_dev = std::sqrt(variance);
        
        return BenchmarkResult{
            name, mean, std_dev, min_time, max_time, times.size(), {}
        };
    }
    
    void export_json(const std::string& filename) const {
        std::ofstream file(filename);
        file << "{\n";
        file << "  \"benchmark_suite\": \"Active Inference Sim Lab\",\n";
        file << "  \"timestamp\": \"" << get_iso_timestamp() << "\",\n";
        file << "  \"config\": {\n";
        file << "    \"warmup_iterations\": " << config_.warmup_iterations << ",\n";
        file << "    \"benchmark_iterations\": " << config_.benchmark_iterations << ",\n";
        file << "    \"time_limit_seconds\": " << config_.time_limit_seconds << "\n";
        file << "  },\n";
        file << "  \"results\": [\n";
        
        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& result = results_[i];
            file << "    {\n";
            file << "      \"name\": \"" << result.name << "\",\n";
            file << "      \"mean_time_ms\": " << result.mean_time_ms << ",\n";
            file << "      \"std_dev_ms\": " << result.std_dev_ms << ",\n";
            file << "      \"min_time_ms\": " << result.min_time_ms << ",\n";
            file << "      \"max_time_ms\": " << result.max_time_ms << ",\n";
            file << "      \"iterations\": " << result.iterations << "\n";
            file << "    }";
            if (i < results_.size() - 1) file << ",";
            file << "\n";
        }
        
        file << "  ]\n";
        file << "}\n";
    }
    
    void export_csv(const std::string& filename) const {
        std::ofstream file(filename);
        file << "name,mean_time_ms,std_dev_ms,min_time_ms,max_time_ms,iterations\n";
        
        for (const auto& result : results_) {
            file << result.name << ","
                 << result.mean_time_ms << ","
                 << result.std_dev_ms << ","
                 << result.min_time_ms << ","
                 << result.max_time_ms << ","
                 << result.iterations << "\n";
        }
    }
    
    void export_human_readable(const std::string& filename) const {
        std::ofstream file(filename);
        file << "Active Inference Sim Lab - Benchmark Results\n";
        file << "==========================================\n\n";
        
        for (const auto& result : results_) {
            file << "Benchmark: " << result.name << "\n";
            file << "  Mean time: " << std::fixed << std::setprecision(3) 
                 << result.mean_time_ms << " ms\n";
            file << "  Std dev:   " << std::fixed << std::setprecision(3) 
                 << result.std_dev_ms << " ms\n";
            file << "  Min time:  " << std::fixed << std::setprecision(3) 
                 << result.min_time_ms << " ms\n";
            file << "  Max time:  " << std::fixed << std::setprecision(3) 
                 << result.max_time_ms << " ms\n";
            file << "  Iterations: " << result.iterations << "\n\n";
        }
    }
    
    std::string get_iso_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        return ss.str();
    }
};

// Convenience macros for benchmarking
#define BENCHMARK(suite, name, code) \
    suite.add_benchmark(name, [&]() { code; })

#define BENCHMARK_WITH_SETUP(suite, name, setup_code, benchmark_code) \
    suite.add_benchmark(name, [&]() { \
        setup_code; \
        benchmark_code; \
    })

} // namespace benchmarks
} // namespace active_inference
```

Create `cpp/benchmarks/core_benchmarks.cpp`:

```cpp
#include "benchmark_framework.hpp"
#include "../src/core/free_energy.hpp"
#include "../src/core/belief_state.hpp"
#include "../src/inference/variational_inference.hpp"
#include <random>
#include <vector>

using namespace active_inference::benchmarks;

// Test data generation utilities
std::vector<double> generate_random_vector(size_t size, double mean = 0.0, double std_dev = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, std_dev);
    
    std::vector<double> result(size);
    for (auto& value : result) {
        value = dist(gen);
    }
    return result;
}

std::vector<std::vector<double>> generate_random_matrix(size_t rows, size_t cols) {
    std::vector<std::vector<double>> matrix(rows);
    for (auto& row : matrix) {
        row = generate_random_vector(cols);
    }
    return matrix;
}

int main() {
    BenchmarkConfig config;
    config.benchmark_iterations = 1000;
    config.warmup_iterations = 100;
    config.output_format = "json";
    
    BenchmarkSuite suite(config);
    
    // Free Energy Computation Benchmarks
    {
        auto observations = generate_random_vector(10);
        auto beliefs = generate_random_vector(10);
        auto precision_matrix = generate_random_matrix(10, 10);
        
        BENCHMARK(suite, "free_energy_small_state", {
            auto fe = active_inference::compute_free_energy(observations, beliefs, precision_matrix);
        });
    }
    
    {
        auto observations = generate_random_vector(100);
        auto beliefs = generate_random_vector(100);
        auto precision_matrix = generate_random_matrix(100, 100);
        
        BENCHMARK(suite, "free_energy_medium_state", {
            auto fe = active_inference::compute_free_energy(observations, beliefs, precision_matrix);
        });
    }
    
    {
        auto observations = generate_random_vector(1000);
        auto beliefs = generate_random_vector(1000);
        auto precision_matrix = generate_random_matrix(1000, 1000);
        
        BENCHMARK(suite, "free_energy_large_state", {
            auto fe = active_inference::compute_free_energy(observations, beliefs, precision_matrix);
        });
    }
    
    // Belief State Update Benchmarks
    {
        active_inference::BeliefState belief_state(10);
        auto observations = generate_random_vector(10);
        
        BENCHMARK(suite, "belief_update_small", {
            belief_state.update(observations);
        });
    }
    
    {
        active_inference::BeliefState belief_state(100);
        auto observations = generate_random_vector(100);
        
        BENCHMARK(suite, "belief_update_medium", {
            belief_state.update(observations);
        });
    }
    
    // Variational Inference Benchmarks
    {
        active_inference::VariationalInference vi(10, 10);
        auto observations = generate_random_vector(10);
        
        BENCHMARK(suite, "variational_inference_small", {
            vi.infer(observations);
        });
    }
    
    {
        active_inference::VariationalInference vi(100, 100);
        auto observations = generate_random_vector(100);
        
        BENCHMARK(suite, "variational_inference_medium", {
            vi.infer(observations);
        });
    }
    
    // Memory allocation benchmarks
    BENCHMARK(suite, "vector_allocation_small", {
        auto vec = generate_random_vector(100);
    });
    
    BENCHMARK(suite, "vector_allocation_large", {
        auto vec = generate_random_vector(10000);
    });
    
    BENCHMARK(suite, "matrix_allocation_small", {
        auto matrix = generate_random_matrix(10, 10);
    });
    
    BENCHMARK(suite, "matrix_allocation_medium", {
        auto matrix = generate_random_matrix(100, 100);
    });
    
    // Run all benchmarks
    suite.run_all();
    
    // Export results
    suite.export_results("benchmark_results.json");
    
    std::cout << "\nResults exported to benchmark_results.json" << std::endl;
    
    return 0;
}
```

## 2. Python Benchmark Framework

Create `src/python/active_inference/benchmarks/benchmark_suite.py`:

```python
"""
Python benchmark suite for Active Inference Sim Lab
"""
import time
import json
import csv
import statistics
import functools
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import multiprocessing as mp

@dataclass
class BenchmarkResult:
    name: str
    mean_time_ms: float
    std_dev_ms: float
    min_time_ms: float
    max_time_ms: float
    iterations: int
    custom_metrics: Dict[str, float] = None

@dataclass
class BenchmarkConfig:
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    time_limit_seconds: float = 60.0
    auto_adjust_iterations: bool = True
    output_format: str = "json"  # json, csv, human
    parallel_execution: bool = False
    num_processes: Optional[int] = None

class BenchmarkSuite:
    """Comprehensive Python benchmark suite"""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.benchmarks: List[tuple[str, Callable]] = []
        self.results: List[BenchmarkResult] = []
    
    def add_benchmark(self, name: str, benchmark_func: Callable):
        """Register a benchmark function"""
        self.benchmarks.append((name, benchmark_func))
    
    def benchmark(self, name: str = None):
        """Decorator for registering benchmark functions"""
        def decorator(func):
            benchmark_name = name or func.__name__
            self.add_benchmark(benchmark_name, func)
            return func
        return decorator
    
    def run_all(self) -> List[BenchmarkResult]:
        """Run all registered benchmarks"""
        self.results.clear()
        
        print(f"Running {len(self.benchmarks)} benchmarks...")
        
        if self.config.parallel_execution:
            self._run_parallel()
        else:
            self._run_sequential()
        
        return self.results
    
    def _run_sequential(self):
        """Run benchmarks sequentially"""
        for name, func in self.benchmarks:
            print(f"Benchmarking: {name}...", end=" ", flush=True)
            result = self._run_single_benchmark(name, func)
            self.results.append(result)
            print(f"{result.mean_time_ms:.3f} ms")
    
    def _run_parallel(self):
        """Run benchmarks in parallel"""
        num_processes = self.config.num_processes or mp.cpu_count()
        
        with mp.Pool(num_processes) as pool:
            tasks = [
                pool.apply_async(self._run_single_benchmark, (name, func))
                for name, func in self.benchmarks
            ]
            
            for i, task in enumerate(tasks):
                result = task.get()
                self.results.append(result)
                print(f"Completed: {result.name} - {result.mean_time_ms:.3f} ms")
    
    def _run_single_benchmark(self, name: str, func: Callable) -> BenchmarkResult:
        """Run a single benchmark function"""
        times = []
        
        # Warmup phase
        for _ in range(self.config.warmup_iterations):
            func()
        
        # Benchmark phase
        start_time = time.time()
        
        for i in range(self.config.benchmark_iterations):
            iteration_start = time.perf_counter()
            func()
            iteration_end = time.perf_counter()
            
            times.append((iteration_end - iteration_start) * 1000)  # Convert to ms
            
            # Check time limit
            if self.config.auto_adjust_iterations:
                elapsed = time.time() - start_time
                if elapsed > self.config.time_limit_seconds:
                    break
        
        # Calculate statistics
        mean_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        min_time = min(times)
        max_time = max(times)
        
        return BenchmarkResult(
            name=name,
            mean_time_ms=mean_time,
            std_dev_ms=std_dev,
            min_time_ms=min_time,
            max_time_ms=max_time,
            iterations=len(times)
        )
    
    def export_results(self, filename: str):
        """Export benchmark results"""
        if self.config.output_format == "json":
            self._export_json(filename)
        elif self.config.output_format == "csv":
            self._export_csv(filename)
        else:
            self._export_human_readable(filename)
    
    def _export_json(self, filename: str):
        """Export results as JSON"""
        data = {
            "benchmark_suite": "Active Inference Sim Lab Python",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "results": [asdict(result) for result in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_csv(self, filename: str):
        """Export results as CSV"""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'name', 'mean_time_ms', 'std_dev_ms', 
                'min_time_ms', 'max_time_ms', 'iterations'
            ])
            
            for result in self.results:
                writer.writerow([
                    result.name, result.mean_time_ms, result.std_dev_ms,
                    result.min_time_ms, result.max_time_ms, result.iterations
                ])
    
    def _export_human_readable(self, filename: str):
        """Export results in human-readable format"""
        with open(filename, 'w') as f:
            f.write("Active Inference Sim Lab - Python Benchmark Results\n")
            f.write("=" * 50 + "\n\n")
            
            for result in self.results:
                f.write(f"Benchmark: {result.name}\n")
                f.write(f"  Mean time: {result.mean_time_ms:.3f} ms\n")
                f.write(f"  Std dev:   {result.std_dev_ms:.3f} ms\n")
                f.write(f"  Min time:  {result.min_time_ms:.3f} ms\n")
                f.write(f"  Max time:  {result.max_time_ms:.3f} ms\n")
                f.write(f"  Iterations: {result.iterations}\n\n")
    
    def compare_with_baseline(self, baseline_file: str, threshold: float = 0.1):
        """Compare current results with baseline performance"""
        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
            
            baseline_results = {
                result['name']: result['mean_time_ms'] 
                for result in baseline_data['results']
            }
            
            regressions = []
            improvements = []
            
            for result in self.results:
                if result.name in baseline_results:
                    baseline_time = baseline_results[result.name]
                    current_time = result.mean_time_ms
                    
                    change_ratio = (current_time - baseline_time) / baseline_time
                    
                    if change_ratio > threshold:
                        regressions.append((result.name, change_ratio))
                    elif change_ratio < -threshold:
                        improvements.append((result.name, abs(change_ratio)))
            
            return {
                'regressions': regressions,
                'improvements': improvements,
                'threshold': threshold
            }
            
        except FileNotFoundError:
            print(f"Baseline file {baseline_file} not found")
            return None

# Utility decorators for benchmark functions
def measure_memory(func):
    """Decorator to measure memory usage during benchmark"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import tracemalloc
        tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Store memory metrics (would need to be handled by benchmark framework)
        if hasattr(wrapper, '_memory_stats'):
            wrapper._memory_stats = {'current': current, 'peak': peak}
        
        return result
    return wrapper

def parametrized_benchmark(*parameters):
    """Decorator for parametrized benchmarks"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(suite):
            for param in parameters:
                param_name = f"{func.__name__}_{param}"
                suite.add_benchmark(param_name, lambda: func(param))
        return wrapper
    return decorator
```

Create `src/python/active_inference/benchmarks/core_benchmarks.py`:

```python
"""
Core benchmarks for Active Inference Python components
"""
import numpy as np
from ..benchmarks.benchmark_suite import BenchmarkSuite, BenchmarkConfig
import sys
import os

# Add the active_inference module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

def generate_test_data(size, seed=42):
    """Generate reproducible test data"""
    np.random.seed(seed)
    return np.random.randn(size)

def generate_test_matrix(rows, cols, seed=42):
    """Generate reproducible test matrix"""
    np.random.seed(seed)
    return np.random.randn(rows, cols)

def main():
    config = BenchmarkConfig(
        benchmark_iterations=1000,
        warmup_iterations=100,
        output_format="json"
    )
    
    suite = BenchmarkSuite(config)
    
    # NumPy operations benchmarks
    @suite.benchmark("numpy_small_matrix_multiply")
    def numpy_small_matmul():
        a = generate_test_matrix(10, 10)
        b = generate_test_matrix(10, 10)
        return np.dot(a, b)
    
    @suite.benchmark("numpy_medium_matrix_multiply")
    def numpy_medium_matmul():
        a = generate_test_matrix(100, 100)
        b = generate_test_matrix(100, 100)
        return np.dot(a, b)
    
    @suite.benchmark("numpy_large_matrix_multiply")
    def numpy_large_matmul():
        a = generate_test_matrix(500, 500)
        b = generate_test_matrix(500, 500)
        return np.dot(a, b)
    
    # Belief state operations
    @suite.benchmark("belief_state_update_small")
    def belief_update_small():
        beliefs = generate_test_data(10)
        observations = generate_test_data(10)
        precision = generate_test_matrix(10, 10)
        
        # Simulate belief update
        error = observations - beliefs
        update = np.dot(precision, error)
        return beliefs + 0.1 * update
    
    @suite.benchmark("belief_state_update_medium")
    def belief_update_medium():
        beliefs = generate_test_data(100)
        observations = generate_test_data(100)
        precision = generate_test_matrix(100, 100)
        
        error = observations - beliefs
        update = np.dot(precision, error)
        return beliefs + 0.1 * update
    
    @suite.benchmark("belief_state_update_large")
    def belief_update_large():
        beliefs = generate_test_data(1000)
        observations = generate_test_data(1000)
        precision = generate_test_matrix(1000, 1000)
        
        error = observations - beliefs
        update = np.dot(precision, error)
        return beliefs + 0.1 * update
    
    # Free energy computation benchmarks
    @suite.benchmark("free_energy_small")
    def free_energy_small():
        observations = generate_test_data(10)
        beliefs = generate_test_data(10)
        precision = generate_test_matrix(10, 10)
        
        error = observations - beliefs
        accuracy = -0.5 * np.dot(error.T, np.dot(precision, error))
        complexity = -0.5 * np.sum(beliefs**2)
        return accuracy + complexity
    
    @suite.benchmark("free_energy_medium")
    def free_energy_medium():
        observations = generate_test_data(100)
        beliefs = generate_test_data(100)
        precision = generate_test_matrix(100, 100)
        
        error = observations - beliefs
        accuracy = -0.5 * np.dot(error.T, np.dot(precision, error))
        complexity = -0.5 * np.sum(beliefs**2)
        return accuracy + complexity
    
    # Eigenvalue decomposition (common in active inference)
    @suite.benchmark("eigenvalue_decomp_small")
    def eigenval_small():
        matrix = generate_test_matrix(50, 50)
        symmetric_matrix = np.dot(matrix.T, matrix)
        return np.linalg.eigh(symmetric_matrix)
    
    @suite.benchmark("eigenvalue_decomp_medium")
    def eigenval_medium():
        matrix = generate_test_matrix(200, 200)
        symmetric_matrix = np.dot(matrix.T, matrix)
        return np.linalg.eigh(symmetric_matrix)
    
    # Variational inference simulation
    @suite.benchmark("variational_inference_iteration")
    def variational_iteration():
        # Simulate one iteration of variational inference
        mu = generate_test_data(50)
        sigma = np.abs(generate_test_data(50)) + 0.1
        observations = generate_test_data(50)
        
        # Gradient update simulation
        grad_mu = observations - mu
        grad_sigma = -1.0/sigma + (observations - mu)**2 / sigma**3
        
        # Update parameters
        mu_new = mu + 0.01 * grad_mu
        sigma_new = sigma + 0.01 * grad_sigma
        
        return mu_new, sigma_new
    
    # Memory allocation benchmarks
    @suite.benchmark("memory_allocation_vectors")
    def memory_alloc_vectors():
        vectors = []
        for i in range(100):
            vectors.append(np.random.randn(1000))
        return vectors
    
    @suite.benchmark("memory_allocation_matrices")
    def memory_alloc_matrices():
        matrices = []
        for i in range(10):
            matrices.append(np.random.randn(100, 100))
        return matrices
    
    # Run benchmarks
    print("Starting Python benchmarks...")
    results = suite.run_all()
    
    # Export results
    suite.export_results("python_benchmark_results.json")
    print("\nResults exported to python_benchmark_results.json")
    
    # Display summary
    print("\nBenchmark Summary:")
    print("-" * 50)
    for result in results:
        print(f"{result.name:<40} {result.mean_time_ms:>8.3f} ms")

if __name__ == "__main__":
    main()
```

## 3. Continuous Performance Monitoring

Create `scripts/performance_monitor.py`:

```python
#!/usr/bin/env python3
"""
Continuous performance monitoring script
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

class PerformanceMonitor:
    """Monitors performance trends and detects regressions"""
    
    def __init__(self, baseline_dir: str = "benchmarks/baselines"):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
    def run_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark suites"""
        results = {}
        
        # Run C++ benchmarks
        cpp_result = self._run_cpp_benchmarks()
        if cpp_result:
            results['cpp'] = cpp_result
        
        # Run Python benchmarks
        python_result = self._run_python_benchmarks()
        if python_result:
            results['python'] = python_result
        
        return results
    
    def _run_cpp_benchmarks(self) -> Dict[str, Any]:
        """Run C++ benchmark suite"""
        try:
            # Build benchmarks if needed
            build_dir = Path("build")
            if not (build_dir / "cpp_benchmarks").exists():
                print("Building C++ benchmarks...")
                subprocess.run(["make", "build-cpp"], check=True)
            
            # Run benchmarks
            print("Running C++ benchmarks...")
            result = subprocess.run(
                ["./build/cpp_benchmarks"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Load results
            with open("benchmark_results.json", 'r') as f:
                return json.load(f)
                
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"C++ benchmark failed: {e}")
            return None
    
    def _run_python_benchmarks(self) -> Dict[str, Any]:
        """Run Python benchmark suite"""
        try:
            print("Running Python benchmarks...")
            result = subprocess.run([
                sys.executable, "-m", "active_inference.benchmarks.core_benchmarks"
            ], capture_output=True, text=True, check=True)
            
            # Load results
            with open("python_benchmark_results.json", 'r') as f:
                return json.load(f)
                
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Python benchmark failed: {e}")
            return None
    
    def detect_regressions(self, current_results: Dict[str, Any], 
                          threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Detect performance regressions"""
        regressions = []
        
        for suite_name, suite_results in current_results.items():
            baseline_file = self.baseline_dir / f"{suite_name}_baseline.json"
            
            if not baseline_file.exists():
                print(f"No baseline found for {suite_name}, creating one...")
                self._save_baseline(suite_name, suite_results)
                continue
            
            # Load baseline
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)
            
            # Compare results
            baseline_lookup = {
                result['name']: result['mean_time_ms']
                for result in baseline['results']
            }
            
            for result in suite_results['results']:
                name = result['name']
                current_time = result['mean_time_ms']
                
                if name in baseline_lookup:
                    baseline_time = baseline_lookup[name]
                    regression_ratio = (current_time - baseline_time) / baseline_time
                    
                    if regression_ratio > threshold:
                        regressions.append({
                            'suite': suite_name,
                            'benchmark': name,
                            'current_time_ms': current_time,
                            'baseline_time_ms': baseline_time,
                            'regression_ratio': regression_ratio,
                            'regression_percent': regression_ratio * 100
                        })
        
        return regressions
    
    def _save_baseline(self, suite_name: str, results: Dict[str, Any]):
        """Save benchmark results as new baseline"""
        baseline_file = self.baseline_dir / f"{suite_name}_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def update_baselines(self, results: Dict[str, Any]):
        """Update baseline files with current results"""
        for suite_name, suite_results in results.items():
            self._save_baseline(suite_name, suite_results)
        print("Baselines updated successfully")
    
    def generate_report(self, results: Dict[str, Any], 
                       regressions: List[Dict[str, Any]]) -> str:
        """Generate performance report"""
        report = []
        report.append("# Performance Benchmark Report")
        report.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
        report.append("")
        
        # Summary
        total_benchmarks = sum(len(suite['results']) for suite in results.values())
        report.append(f"**Total Benchmarks:** {total_benchmarks}")
        report.append(f"**Performance Regressions:** {len(regressions)}")
        report.append("")
        
        # Regressions
        if regressions:
            report.append("## ⚠️ Performance Regressions Detected")
            report.append("")
            for regression in regressions:
                report.append(f"- **{regression['suite']}/{regression['benchmark']}**")
                report.append(f"  - Current: {regression['current_time_ms']:.3f} ms")
                report.append(f"  - Baseline: {regression['baseline_time_ms']:.3f} ms")
                report.append(f"  - Regression: {regression['regression_percent']:.1f}%")
                report.append("")
        else:
            report.append("## ✅ No Performance Regressions Detected")
            report.append("")
        
        # Detailed results
        for suite_name, suite_results in results.items():
            report.append(f"## {suite_name.upper()} Benchmark Results")
            report.append("")
            report.append("| Benchmark | Mean Time (ms) | Std Dev (ms) | Iterations |")
            report.append("|-----------|----------------|--------------|------------|")
            
            for result in suite_results['results']:
                report.append(
                    f"| {result['name']} | {result['mean_time_ms']:.3f} | "
                    f"{result['std_dev_ms']:.3f} | {result['iterations']} |"
                )
            report.append("")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Performance monitoring script")
    parser.add_argument("--update-baseline", action="store_true",
                       help="Update baseline performance metrics")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Regression detection threshold (default: 0.1 = 10%)")
    parser.add_argument("--output", default="performance_report.md",
                       help="Output file for performance report")
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor()
    
    # Run benchmarks
    print("Running performance benchmarks...")
    results = monitor.run_benchmarks()
    
    if not results:
        print("No benchmark results available")
        return 1
    
    # Update baselines if requested
    if args.update_baseline:
        monitor.update_baselines(results)
        return 0
    
    # Detect regressions
    regressions = monitor.detect_regressions(results, args.threshold)
    
    # Generate report
    report = monitor.generate_report(results, regressions)
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"Performance report saved to {args.output}")
    
    # Exit with error code if regressions detected
    if regressions:
        print(f"⚠️  {len(regressions)} performance regressions detected!")
        return 1
    else:
        print("✅ No performance regressions detected")
        return 0

if __name__ == "__main__":
    sys.exit(main())
```

## 4. Integration with CI/CD

This performance monitoring integrates with the GitHub Actions workflow for:

- **Automated Performance Testing**: Run benchmarks on every PR
- **Regression Detection**: Alert on performance degradation
- **Baseline Management**: Track performance over time
- **Performance Reporting**: Generate detailed performance reports

The benchmark framework provides comprehensive performance monitoring for both C++ and Python components, ensuring optimal performance throughout development.