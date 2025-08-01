#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) Generator
Generates SPDX and CycloneDX format SBOMs for Active Inference Simulation Lab
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any
import uuid
import hashlib

class SBOMGenerator:
    """Generate Software Bill of Materials in multiple formats."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.document_id = str(uuid.uuid4())
        
    def generate_spdx(self) -> Dict[str, Any]:
        """Generate SPDX format SBOM."""
        dependencies = self._get_dependencies()
        
        spdx_document = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": f"SPDXRef-DOCUMENT-{self.document_id}",
            "documentName": "active-inference-sim-lab",
            "documentNamespace": f"https://terragonlabs.com/sbom/{self.document_id}",
            "creationInfo": {
                "created": self.timestamp,
                "creators": ["Tool: Terragon SBOM Generator"],
                "licenseListVersion": "3.20"
            },
            "packages": self._create_spdx_packages(dependencies),
            "relationships": self._create_spdx_relationships(dependencies)
        }
        
        return spdx_document
    
    def generate_cyclonedx(self) -> Dict[str, Any]:
        """Generate CycloneDX format SBOM."""
        dependencies = self._get_dependencies()
        
        cyclonedx_document = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": f"urn:uuid:{self.document_id}",
            "version": 1,
            "metadata": {
                "timestamp": self.timestamp,
                "tools": [
                    {
                        "vendor": "Terragon Labs",
                        "name": "SBOM Generator",
                        "version": "1.0.0"
                    }
                ],
                "component": {
                    "type": "application",
                    "bom-ref": "active-inference-sim-lab",
                    "name": "active-inference-sim-lab",
                    "version": "0.1.0",
                    "description": "Lightweight toolkit for building active inference agents",
                    "licenses": [{"license": {"id": "Apache-2.0"}}]
                }
            },
            "components": self._create_cyclonedx_components(dependencies),
            "dependencies": self._create_cyclonedx_dependencies(dependencies)
        }
        
        return cyclonedx_document
    
    def _get_dependencies(self) -> List[Dict[str, Any]]:
        """Extract dependencies from various sources."""
        dependencies = []
        
        # Python dependencies
        python_deps = self._get_python_dependencies()
        dependencies.extend(python_deps)
        
        # C++ dependencies (from CMakeLists.txt)
        cpp_deps = self._get_cpp_dependencies()
        dependencies.extend(cpp_deps)
        
        # System dependencies
        system_deps = self._get_system_dependencies()
        dependencies.extend(system_deps)
        
        return dependencies
    
    def _get_python_dependencies(self) -> List[Dict[str, Any]]:
        """Get Python package dependencies."""
        dependencies = []
        
        # Read from requirements files
        req_files = ["requirements.txt", "requirements-dev.txt"]
        
        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                with open(req_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            dep = self._parse_requirement(line)
                            if dep:
                                dependencies.append(dep)
        
        # Also check pyproject.toml
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            # Parse pyproject.toml dependencies
            deps = self._parse_pyproject_dependencies(pyproject_path)
            dependencies.extend(deps)
        
        return dependencies
    
    def _get_cpp_dependencies(self) -> List[Dict[str, Any]]:
        """Get C++ dependencies from CMakeLists.txt."""
        dependencies = []
        
        cmake_path = self.project_root / "CMakeLists.txt"
        if cmake_path.exists():
            with open(cmake_path) as f:
                content = f.read()
                
                # Look for find_package calls
                import re
                find_packages = re.findall(r'find_package\((\w+)', content)
                
                for package in find_packages:
                    dependencies.append({
                        "name": package.lower(),
                        "version": "unknown",
                        "type": "library",
                        "ecosystem": "cpp"
                    })
        
        return dependencies
    
    def _get_system_dependencies(self) -> List[Dict[str, Any]]:
        """Get system-level dependencies."""
        dependencies = []
        
        # Docker base images
        dockerfile_path = self.project_root / "Dockerfile"
        if dockerfile_path.exists():
            with open(dockerfile_path) as f:
                for line in f:
                    if line.strip().startswith("FROM"):
                        base_image = line.strip().split()[1]
                        dependencies.append({
                            "name": base_image.split(":")[0],
                            "version": base_image.split(":")[1] if ":" in base_image else "latest",
                            "type": "container",
                            "ecosystem": "docker"
                        })
        
        return dependencies
    
    def _parse_requirement(self, requirement: str) -> Dict[str, Any]:
        """Parse a Python requirement string."""
        import re
        
        # Handle basic version specifiers
        match = re.match(r'^([a-zA-Z0-9_-]+)([><=!]*)([\d.]*)', requirement)
        if match:
            name, operator, version = match.groups()
            return {
                "name": name,
                "version": version or "unknown",
                "type": "library", 
                "ecosystem": "python",
                "version_constraint": f"{operator}{version}" if operator else None
            }
        
        return None
    
    def _parse_pyproject_dependencies(self, pyproject_path: Path) -> List[Dict[str, Any]]:
        """Parse dependencies from pyproject.toml."""
        dependencies = []
        
        try:
            import tomllib
        except ImportError:
            # Fallback for Python < 3.11
            try:
                import tomli as tomllib
            except ImportError:
                return dependencies
        
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            
            project_deps = data.get("project", {}).get("dependencies", [])
            for dep in project_deps:
                parsed = self._parse_requirement(dep)
                if parsed:
                    dependencies.append(parsed)
        
        return dependencies
    
    def _create_spdx_packages(self, dependencies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create SPDX package entries."""
        packages = []
        
        for dep in dependencies:
            package = {
                "SPDXID": f"SPDXRef-Package-{dep['name'].replace('-', '')}",
                "name": dep["name"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "copyrightText": "NOASSERTION"
            }
            
            if dep.get("version") != "unknown":
                package["versionInfo"] = dep["version"]
            
            packages.append(package)
        
        return packages
    
    def _create_spdx_relationships(self, dependencies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create SPDX relationship entries."""
        relationships = []
        
        for dep in dependencies:
            relationships.append({
                "spdxElementId": f"SPDXRef-DOCUMENT-{self.document_id}",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": f"SPDXRef-Package-{dep['name'].replace('-', '')}"
            })
        
        return relationships
    
    def _create_cyclonedx_components(self, dependencies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create CycloneDX component entries."""
        components = []
        
        for dep in dependencies:
            component = {
                "type": "library",
                "bom-ref": f"{dep['ecosystem']}/{dep['name']}@{dep.get('version', 'unknown')}",
                "name": dep["name"],
                "version": dep.get("version", "unknown"),
                "scope": "required"
            }
            
            # Add PURL (Package URL) if possible
            if dep["ecosystem"] == "python":
                component["purl"] = f"pkg:pypi/{dep['name']}@{dep.get('version', 'unknown')}"
            elif dep["ecosystem"] == "docker":
                component["purl"] = f"pkg:docker/{dep['name']}@{dep.get('version', 'latest')}"
            
            components.append(component)
        
        return components
    
    def _create_cyclonedx_dependencies(self, dependencies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create CycloneDX dependency entries."""
        deps = [{
            "ref": "active-inference-sim-lab",
            "dependsOn": [
                f"{dep['ecosystem']}/{dep['name']}@{dep.get('version', 'unknown')}"
                for dep in dependencies
            ]
        }]
        
        return deps
    
    def save_sboms(self, output_dir: Path):
        """Save SBOMs to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Generate and save SPDX
        spdx_sbom = self.generate_spdx()
        with open(output_dir / "sbom.spdx.json", "w") as f:
            json.dump(spdx_sbom, f, indent=2)
        
        # Generate and save CycloneDX
        cyclonedx_sbom = self.generate_cyclonedx()
        with open(output_dir / "sbom.cyclonedx.json", "w") as f:
            json.dump(cyclonedx_sbom, f, indent=2)
        
        print(f"SBOMs generated in {output_dir}")
        print(f"- SPDX: sbom.spdx.json ({len(spdx_sbom['packages'])} packages)")
        print(f"- CycloneDX: sbom.cyclonedx.json ({len(cyclonedx_sbom['components'])} components)")

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SBOM for Active Inference Simulation Lab")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output-dir", default="./security/sbom", help="Output directory for SBOMs")
    
    args = parser.parse_args()
    
    generator = SBOMGenerator(args.project_root)
    generator.save_sboms(args.output_dir)

if __name__ == "__main__":
    main()