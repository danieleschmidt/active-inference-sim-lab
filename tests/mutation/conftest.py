# Mutation Testing Configuration
# Supports mutmut for comprehensive test quality assessment

import pytest
import os
from pathlib import Path

# Mutation testing configuration
MUTATION_CONFIG = {
    "source_dirs": ["src/python/active_inference"],
    "test_dirs": ["tests/unit", "tests/integration"],
    "exclude_patterns": [
        "*/tests/*",
        "*/__pycache__/*", 
        "*/migrations/*",
        "*/venv/*",
        "*/build/*"
    ],
    "mutation_threshold": 80,  # Minimum mutation score
    "timeout_factor": 2.0      # Test timeout multiplier
}

@pytest.fixture(scope="session")
def mutation_config():
    """Provide mutation testing configuration."""
    return MUTATION_CONFIG

@pytest.fixture(scope="session") 
def source_files():
    """Get list of source files for mutation testing."""
    source_files = []
    for source_dir in MUTATION_CONFIG["source_dirs"]:
        if Path(source_dir).exists():
            for py_file in Path(source_dir).rglob("*.py"):
                if not any(pattern in str(py_file) for pattern in MUTATION_CONFIG["exclude_patterns"]):
                    source_files.append(str(py_file))
    return source_files

@pytest.fixture
def mutation_runner():
    """Mutation testing runner fixture."""
    class MutationRunner:
        def run_mutations(self, target_file=None):
            """Run mutation tests on target file or all files."""
            import subprocess
            cmd = ["mutmut", "run"]
            if target_file:
                cmd.extend(["--paths-to-mutate", target_file])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
            
        def get_mutation_score(self):
            """Get current mutation testing score."""
            import subprocess
            result = subprocess.run(
                ["mutmut", "results"], 
                capture_output=True, 
                text=True
            )
            # Parse mutation score from output
            if "killed" in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "killed" in line and "/" in line:
                        parts = line.split()
                        killed = int(parts[0])
                        total = int(parts[2])
                        return (killed / total) * 100 if total > 0 else 0
            return 0
    
    return MutationRunner()