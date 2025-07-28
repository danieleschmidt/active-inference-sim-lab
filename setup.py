#!/usr/bin/env python3
"""
Setup script for active-inference-sim-lab.

This setup.py is mainly for building C++ extensions with pybind11.
Most configuration is in pyproject.toml.
"""

import os
import sys
from pathlib import Path

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup

# Check if CMake is available
try:
    import cmake
    CMAKE_AVAILABLE = True
except ImportError:
    CMAKE_AVAILABLE = False

# The main interface is through Pybind11Extension.
# * You can add cxx_std=14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

# Define C++ extension modules
cpp_sources = [
    "cpp/src/core/free_energy.cpp",
    "cpp/src/core/generative_model.cpp",
    "cpp/src/inference/belief_updater.cpp",
    "cpp/src/inference/variational_inference.cpp",
    "cpp/src/planning/active_planner.cpp",
    "cpp/src/planning/trajectory_optimizer.cpp",
    "cpp/src/utils/math_utils.cpp",
    "cpp/src/utils/random_utils.cpp",
    "cpp/src/bindings/python_bindings.cpp",
]

# Check if C++ source files exist
existing_sources = []
for src in cpp_sources:
    if os.path.exists(src):
        existing_sources.append(src)

ext_modules = []

if existing_sources:
    ext_modules = [
        Pybind11Extension(
            "active_inference._core",
            existing_sources,
            include_dirs=[
                "cpp/include",
                "include",
            ],
            language="c++",
            cxx_std=17,
            define_macros=[("VERSION_INFO", '"dev"')],
        ),
    ]

# Custom build_ext to handle C++ compilation
class CustomBuildExt(build_ext):
    """Custom build extension to handle C++ compilation."""
    
    def build_extensions(self):
        """Build C++ extensions with appropriate flags."""
        # Compiler-specific options
        ct = self.compiler.compiler_type
        opts = []
        link_opts = []
        
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append("-std=c++17")
            opts.append("-O3")
            opts.append("-ffast-math")
            opts.append("-march=native")
            if sys.platform == "darwin":
                opts.append("-stdlib=libc++")
                opts.append("-mmacosx-version-min=10.14")
                link_opts.append("-mmacosx-version-min=10.14")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
            opts.append("/std:c++17")
            opts.append("/O2")
            opts.append("/arch:AVX2")
        
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        
        super().build_extensions()

if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": CustomBuildExt},
        zip_safe=False,
        python_requires=">=3.9",
    )