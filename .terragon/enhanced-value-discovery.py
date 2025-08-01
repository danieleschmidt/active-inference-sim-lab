#!/usr/bin/env python3
"""
Enhanced Value Discovery Engine for Maturing Repositories
Implements advanced WSJF + ICE + Technical Debt scoring with continuous learning
"""

import json
import subprocess
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple
import hashlib
import statistics

class EnhancedValueDiscovery:
    """Advanced value discovery with machine learning-enhanced scoring."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.metrics_file = self.project_root / ".terragon" / "value-metrics.json"
        self.learning_file = self.project_root / ".terragon" / "learning-model.json"
        
        # Load previous learning data
        self.learning_data = self._load_learning_data()
        
        # Adaptive weights based on repository maturity and learning
        self.weights = self._get_adaptive_weights()
        
        # Value thresholds
        self.thresholds = {
            "min_score": 15.0,
            "security_boost": 2.5,
            "compliance_boost": 1.8,
            "performance_critical": 1.5,
            "automation_value": 1.3
        }
    
    def _load_learning_data(self) -> Dict[str, Any]:
        """Load machine learning model from previous executions."""
        if self.learning_file.exists():
            with open(self.learning_file) as f:
                return json.load(f)
        
        return {
            "estimation_accuracy": [],
            "category_weights": {},
            "success_patterns": [],
            "failure_indicators": [],
            "execution_history": []
        }
    
    def _get_adaptive_weights(self) -> Dict[str, float]:
        """Get adaptive weights based on repository maturity and learning."""
        base_weights = {
            "wsjf": 0.4,
            "ice": 0.25, 
            "technical_debt": 0.25,
            "security": 0.1
        }
        
        # Adjust based on learning data
        if self.learning_data.get("category_weights"):
            for category, weight in self.learning_data["category_weights"].items():
                if category in base_weights:
                    base_weights[category] = weight
        
        return base_weights
    
    def discover_value_opportunities(self) -> List[Dict[str, Any]]:
        """Discover all value opportunities using advanced analysis."""
        opportunities = []
        
        # Security vulnerabilities (highest priority)
        security_items = self._discover_security_vulnerabilities()
        opportunities.extend(security_items)
        
        # Performance optimizations
        performance_items = self._discover_performance_opportunities()
        opportunities.extend(performance_items)
        
        # Technical debt hot-spots
        debt_items = self._discover_technical_debt()
        opportunities.extend(debt_items)
        
        # Automation opportunities
        automation_items = self._discover_automation_opportunities()
        opportunities.extend(automation_items)
        
        # Quality improvements
        quality_items = self._discover_quality_improvements()
        opportunities.extend(quality_items)
        
        # Documentation gaps
        doc_items = self._discover_documentation_gaps()
        opportunities.extend(doc_items)
        
        # Score and rank all opportunities
        scored_opportunities = []
        for item in opportunities:
            score = self._calculate_composite_score(item)
            item["composite_score"] = score
            scored_opportunities.append(item)
        
        # Sort by composite score (descending)
        scored_opportunities.sort(key=lambda x: x["composite_score"], reverse=True)
        
        return scored_opportunities
    
    def _discover_security_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Discover security vulnerabilities and compliance issues."""
        items = []
        
        # Check for outdated dependencies
        try:
            result = subprocess.run(
                ["python3", "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True, text=True, cwd=self.project_root
            )
            
            if result.returncode == 0 and result.stdout.strip():
                outdated_deps = json.loads(result.stdout)
                if outdated_deps:
                    items.append({
                        "id": "SEC-001",
                        "title": "Update vulnerable dependencies",
                        "category": "security",
                        "description": f"Found {len(outdated_deps)} outdated packages that may contain security vulnerabilities",
                        "effort_hours": min(2 + len(outdated_deps) * 0.1, 8),
                        "impact": "high",
                        "urgency": "high",
                        "confidence": 0.9,
                        "risk_reduction": 85,
                        "dependencies": outdated_deps[:5]  # Top 5 for reference
                    })
        except Exception:
            pass
        
        # Check for hardcoded secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]
        
        secret_files = []
        for pattern in ["**/*.py", "**/*.js", "**/*.yaml", "**/*.json"]:
            for file_path in self.project_root.glob(pattern):
                if any(exclude in str(file_path) for exclude in [".git", "node_modules", "__pycache__"]):
                    continue
                    
                try:
                    with open(file_path) as f:
                        content = f.read()
                        for secret_pattern in secret_patterns:
                            if re.search(secret_pattern, content, re.IGNORECASE):
                                secret_files.append(str(file_path.relative_to(self.project_root)))
                                break
                except Exception:
                    continue
        
        if secret_files:
            items.append({
                "id": "SEC-002",
                "title": "Remove hardcoded secrets",
                "category": "security",
                "description": f"Found potential hardcoded secrets in {len(secret_files)} files",
                "effort_hours": len(secret_files) * 0.5,
                "impact": "high",
                "urgency": "high", 
                "confidence": 0.7,
                "risk_reduction": 90,
                "affected_files": secret_files
            })
        
        return items
    
    def _discover_performance_opportunities(self) -> List[Dict[str, Any]]:
        """Discover performance optimization opportunities."""
        items = []
        
        # Look for performance hot-spots in code
        hot_spots = []
        performance_patterns = [
            r'for.*in.*range\(.*\):\s*for.*in.*range',  # Nested loops
            r'\.append\(.*\).*for.*in',  # List comprehension opportunities
            r'time\.sleep\(',  # Blocking sleep calls
            r'requests\.get\(',  # Synchronous HTTP calls
        ]
        
        for py_file in self.project_root.glob("**/*.py"):
            if any(exclude in str(py_file) for exclude in [".git", "__pycache__", "test"]):
                continue
                
            try:
                with open(py_file) as f:
                    content = f.read()
                    for pattern in performance_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            hot_spots.append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "issues": len(matches),
                                "pattern": pattern
                            })
            except Exception:
                continue
        
        if hot_spots:
            total_issues = sum(spot["issues"] for spot in hot_spots)
            items.append({
                "id": "PERF-001",
                "title": "Optimize performance hot-spots",
                "category": "performance",
                "description": f"Found {total_issues} performance optimization opportunities",
                "effort_hours": min(total_issues * 0.5, 16),
                "impact": "medium",
                "urgency": "medium", 
                "confidence": 0.6,
                "value_multiplier": 1.2,
                "hot_spots": hot_spots[:10]  # Top 10
            })
        
        # Database query optimization opportunities
        if (self.project_root / "requirements.txt").exists():
            with open(self.project_root / "requirements.txt") as f:
                deps = f.read()
                if any(db in deps for db in ["psycopg", "sqlalchemy", "django"]):
                    items.append({
                        "id": "PERF-002", 
                        "title": "Optimize database queries",
                        "category": "performance",
                        "description": "Add query optimization and database indexing analysis",
                        "effort_hours": 6,
                        "impact": "high",
                        "urgency": "medium",
                        "confidence": 0.8,
                        "value_multiplier": 1.5
                    })
        
        return items
    
    def _discover_technical_debt(self) -> List[Dict[str, Any]]:
        """Discover technical debt using advanced analysis."""
        items = []
        
        # Count TODO/FIXME items with context analysis
        debt_items = []
        debt_patterns = [
            (r'TODO:?\s*(.+)', "todo"),
            (r'FIXME:?\s*(.+)', "fixme"), 
            (r'XXX:?\s*(.+)', "urgent"),
            (r'HACK:?\s*(.+)', "hack")
        ]
        
        for pattern_files in ["**/*.py", "**/*.js", "**/*.cpp", "**/*.h"]:
            for file_path in self.project_root.glob(pattern_files):
                if any(exclude in str(file_path) for exclude in [".git", "node_modules"]):
                    continue
                    
                try:
                    with open(file_path) as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            for pattern, debt_type in debt_patterns:
                                match = re.search(pattern, line, re.IGNORECASE)
                                if match:
                                    debt_items.append({
                                        "file": str(file_path.relative_to(self.project_root)),
                                        "line": i + 1,
                                        "type": debt_type,
                                        "description": match.group(1).strip(),
                                        "context": line.strip()
                                    })
                except Exception:
                    continue
        
        if debt_items:
            # Categorize debt by severity
            urgent_debt = [item for item in debt_items if item["type"] in ["urgent", "hack"]]
            total_effort = len(debt_items) * 0.5 + len(urgent_debt) * 1.0
            
            items.append({
                "id": "TD-001", 
                "title": f"Address {len(debt_items)} technical debt items",
                "category": "technical_debt",
                "description": f"Resolve technical debt including {len(urgent_debt)} urgent items",
                "effort_hours": min(total_effort, 20),
                "impact": "medium",
                "urgency": "medium" if len(urgent_debt) < 5 else "high",
                "confidence": 0.8,
                "debt_breakdown": {
                    "total": len(debt_items),
                    "urgent": len(urgent_debt),
                    "by_type": {dt: len([i for i in debt_items if i["type"] == dt]) 
                               for dt in ["todo", "fixme", "urgent", "hack"]}
                }
            })
        
        # Code complexity analysis
        complexity_items = self._analyze_code_complexity()
        items.extend(complexity_items)
        
        return items
    
    def _discover_automation_opportunities(self) -> List[Dict[str, Any]]:
        """Discover opportunities for automation improvements."""
        items = []
        
        # Check for missing pre-commit hooks
        if not (self.project_root / ".pre-commit-config.yaml").exists():
            items.append({
                "id": "AUTO-001",
                "title": "Implement comprehensive pre-commit hooks",
                "category": "automation", 
                "description": "Add automated code quality checks before commits",
                "effort_hours": 2,
                "impact": "medium",
                "urgency": "low",
                "confidence": 0.9,
                "value_multiplier": 1.3
            })
        
        # Check for missing GitHub Actions
        github_dir = self.project_root / ".github" / "workflows"
        if github_dir.exists():
            workflow_files = list(github_dir.glob("*.yml")) + list(github_dir.glob("*.yaml"))
            
            expected_workflows = ["ci", "security", "performance", "release"]
            missing_workflows = []
            
            for expected in expected_workflows:
                if not any(expected in wf.stem for wf in workflow_files):
                    missing_workflows.append(expected)
            
            if missing_workflows:
                items.append({
                    "id": "AUTO-002",
                    "title": f"Add missing CI/CD workflows: {', '.join(missing_workflows)}",
                    "category": "automation",
                    "description": f"Implement {len(missing_workflows)} missing workflow automations",
                    "effort_hours": len(missing_workflows) * 2,
                    "impact": "high",
                    "urgency": "medium",
                    "confidence": 0.85,
                    "missing_workflows": missing_workflows
                })
        
        # Dependency update automation
        if not (self.project_root / ".github" / "dependabot.yml").exists():
            items.append({
                "id": "AUTO-003",
                "title": "Setup automated dependency updates",
                "category": "automation",
                "description": "Configure Dependabot for automatic dependency management",
                "effort_hours": 1,
                "impact": "medium",
                "urgency": "low", 
                "confidence": 0.95,
                "value_multiplier": 1.2
            })
        
        return items
    
    def _discover_quality_improvements(self) -> List[Dict[str, Any]]:
        """Discover code quality improvement opportunities."""
        items = []
        
        # Test coverage analysis
        test_files = list(self.project_root.glob("tests/**/*.py"))
        src_files = list(self.project_root.glob("src/**/*.py"))
        
        if src_files:
            test_coverage_ratio = len(test_files) / len(src_files) if src_files else 0
            
            if test_coverage_ratio < 0.8:  # Less than 80% test coverage
                items.append({
                    "id": "QUAL-001",
                    "title": "Improve test coverage",
                    "category": "quality",
                    "description": f"Increase test coverage from {test_coverage_ratio:.1%} to 90%+",
                    "effort_hours": (0.9 - test_coverage_ratio) * 20,
                    "impact": "high",
                    "urgency": "medium",
                    "confidence": 0.8,
                    "current_coverage": f"{test_coverage_ratio:.1%}",
                    "target_coverage": "90%"
                })
        
        # Type hints coverage
        type_hint_files = []
        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file) as f:
                    content = f.read()
                    if "from typing import" in content or "-> " in content:
                        type_hint_files.append(py_file)
            except Exception:
                continue
        
        if src_files:
            type_hint_ratio = len(type_hint_files) / len(src_files)
            if type_hint_ratio < 0.7:  # Less than 70% type hints
                items.append({
                    "id": "QUAL-002",
                    "title": "Add comprehensive type hints",
                    "category": "quality",
                    "description": f"Improve type hint coverage from {type_hint_ratio:.1%} to 90%+",
                    "effort_hours": (0.9 - type_hint_ratio) * 15,
                    "impact": "medium",
                    "urgency": "low",
                    "confidence": 0.7,
                    "current_coverage": f"{type_hint_ratio:.1%}",
                    "target_coverage": "90%"
                })
        
        return items
    
    def _discover_documentation_gaps(self) -> List[Dict[str, Any]]:
        """Discover documentation improvement opportunities."""
        items = []
        
        # API documentation completeness
        api_files = []
        documented_apis = []
        
        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file) as f:
                    content = f.read()
                    # Look for function/class definitions
                    functions = re.findall(r'def\s+(\w+)', content)
                    classes = re.findall(r'class\s+(\w+)', content)
                    
                    if functions or classes:
                        api_files.append(py_file)
                        
                        # Check for docstrings
                        docstring_count = len(re.findall(r'""".*?"""', content, re.DOTALL))
                        if docstring_count > 0:
                            documented_apis.append(py_file)
            except Exception:
                continue
        
        if api_files:
            doc_ratio = len(documented_apis) / len(api_files)
            if doc_ratio < 0.8:  # Less than 80% documented
                items.append({
                    "id": "DOC-001",
                    "title": "Improve API documentation coverage",
                    "category": "documentation",
                    "description": f"Add docstrings and API docs (current: {doc_ratio:.1%})",
                    "effort_hours": (0.9 - doc_ratio) * 10,
                    "impact": "medium",
                    "urgency": "low",
                    "confidence": 0.8,
                    "current_coverage": f"{doc_ratio:.1%}",
                    "target_coverage": "90%"
                })
        
        return items
    
    def _analyze_code_complexity(self) -> List[Dict[str, Any]]:
        """Analyze code complexity and identify refactoring opportunities."""
        items = []
        
        complex_functions = []
        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file) as f:
                    content = f.read()
                    
                    # Simple complexity analysis based on indentation and conditionals
                    functions = re.finditer(r'def\s+(\w+).*?(?=\ndef|\nclass|\Z)', content, re.DOTALL)
                    
                    for match in functions:
                        func_content = match.group(0)
                        
                        # Count complexity indicators
                        if_count = len(re.findall(r'\bif\b', func_content))
                        for_count = len(re.findall(r'\bfor\b', func_content))
                        while_count = len(re.findall(r'\bwhile\b', func_content))
                        try_count = len(re.findall(r'\btry\b', func_content))
                        
                        complexity = if_count + for_count + while_count + try_count
                        
                        if complexity > 8:  # High complexity
                            complex_functions.append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "function": match.group(1),
                                "complexity": complexity
                            })
            except Exception:
                continue
        
        if complex_functions:
            items.append({
                "id": "TD-002",
                "title": f"Refactor {len(complex_functions)} complex functions",
                "category": "technical_debt",
                "description": "Reduce cyclomatic complexity in high-complexity functions",
                "effort_hours": len(complex_functions) * 1.5,
                "impact": "medium",
                "urgency": "low",
                "confidence": 0.7,
                "complex_functions": complex_functions[:10]  # Top 10
            })
        
        return items
    
    def _calculate_composite_score(self, item: Dict[str, Any]) -> float:
        """Calculate composite score using WSJF + ICE + Technical Debt."""
        
        # WSJF Components
        user_value = self._score_user_business_value(item)
        time_criticality = self._score_time_criticality(item)
        risk_reduction = self._score_risk_reduction(item)
        opportunity_enablement = self._score_opportunity_enablement(item)
        
        cost_of_delay = user_value + time_criticality + risk_reduction + opportunity_enablement
        job_size = item.get("effort_hours", 1)
        
        wsjf_score = cost_of_delay / max(job_size, 0.5)
        
        # ICE Components
        impact = self._score_impact(item)
        confidence = item.get("confidence", 0.5) * 10
        ease = self._score_ease(item)
        
        ice_score = impact * confidence * ease
        
        # Technical Debt Score
        debt_score = self._score_technical_debt(item)
        
        # Composite Score with adaptive weights
        composite = (
            self.weights["wsjf"] * self._normalize_score(wsjf_score, 0, 50) +
            self.weights["ice"] * self._normalize_score(ice_score, 0, 1000) + 
            self.weights["technical_debt"] * self._normalize_score(debt_score, 0, 100) +
            self.weights["security"] * self._get_security_boost(item)
        ) * 100
        
        # Apply category-specific multipliers
        multiplier = item.get("value_multiplier", 1.0)
        if item["category"] == "security":
            multiplier *= self.thresholds["security_boost"]
        elif item["category"] == "performance":
            multiplier *= self.thresholds["performance_critical"]
        elif item["category"] == "automation":
            multiplier *= self.thresholds["automation_value"]
        
        return composite * multiplier
    
    def _score_user_business_value(self, item: Dict[str, Any]) -> float:
        """Score user business value (1-10)."""
        impact_map = {"high": 8, "medium": 5, "low": 2}
        base_score = impact_map.get(item.get("impact", "medium"), 5)
        
        # Category-specific adjustments
        if item["category"] == "security":
            return min(base_score + 3, 10)
        elif item["category"] == "performance":
            return min(base_score + 2, 10)
        elif item["category"] == "automation":
            return min(base_score + 1, 10)
        
        return base_score
    
    def _score_time_criticality(self, item: Dict[str, Any]) -> float:
        """Score time criticality (1-10)."""
        urgency_map = {"high": 8, "medium": 5, "low": 2}
        return urgency_map.get(item.get("urgency", "medium"), 5)
    
    def _score_risk_reduction(self, item: Dict[str, Any]) -> float:
        """Score risk reduction (1-10)."""
        risk_reduction = item.get("risk_reduction", 0)
        if risk_reduction > 70:
            return 9
        elif risk_reduction > 40:
            return 6
        elif risk_reduction > 20:
            return 4
        else:
            return 2
    
    def _score_opportunity_enablement(self, item: Dict[str, Any]) -> float:
        """Score opportunity enablement (1-10)."""
        if item["category"] == "automation":
            return 7  # Automation enables future opportunities
        elif item["category"] == "technical_debt":
            return 5  # Debt reduction enables future development
        elif item["category"] == "quality":
            return 4  # Quality improvements enable confidence
        else:
            return 3  # Base opportunity enablement
    
    def _score_impact(self, item: Dict[str, Any]) -> float:
        """Score impact for ICE (1-10)."""
        return self._score_user_business_value(item)
    
    def _score_ease(self, item: Dict[str, Any]) -> float:
        """Score ease of implementation (1-10)."""
        effort = item.get("effort_hours", 8)
        if effort <= 2:
            return 9
        elif effort <= 4:
            return 7
        elif effort <= 8:
            return 5
        elif effort <= 16:
            return 3
        else:
            return 1
    
    def _score_technical_debt(self, item: Dict[str, Any]) -> float:
        """Score technical debt impact (0-100)."""
        if item["category"] == "technical_debt":
            debt_count = item.get("debt_breakdown", {}).get("total", 0)
            urgent_count = item.get("debt_breakdown", {}).get("urgent", 0)
            
            base_score = min(debt_count * 2, 60)
            urgency_bonus = urgent_count * 5
            
            return min(base_score + urgency_bonus, 100)
        
        return 0
    
    def _get_security_boost(self, item: Dict[str, Any]) -> float:
        """Get security priority boost (0-50)."""
        if item["category"] == "security":
            return 50
        else:
            return 0
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-1 range."""
        return max(0, min(1, (score - min_val) / (max_val - min_val)))
    
    def update_learning_model(self, completed_item: Dict[str, Any], 
                            actual_effort: float, actual_impact: float):
        """Update learning model based on completed work."""
        
        predicted_effort = completed_item.get("effort_hours", 0)
        predicted_impact = self._score_impact(completed_item) 
        
        # Calculate accuracy ratios
        effort_ratio = actual_effort / max(predicted_effort, 0.1)
        impact_ratio = actual_impact / max(predicted_impact, 0.1)
        
        # Store learning data
        learning_entry = {
            "timestamp": self.timestamp,
            "item_id": completed_item.get("id"),
            "category": completed_item.get("category"),
            "predicted_effort": predicted_effort,
            "actual_effort": actual_effort,
            "predicted_impact": predicted_impact,
            "actual_impact": actual_impact,
            "effort_accuracy": effort_ratio,
            "impact_accuracy": impact_ratio
        }
        
        self.learning_data["execution_history"].append(learning_entry)
        
        # Update accuracy tracking
        self.learning_data["estimation_accuracy"].append({
            "effort_ratio": effort_ratio,
            "impact_ratio": impact_ratio
        })
        
        # Keep only last 50 entries
        if len(self.learning_data["estimation_accuracy"]) > 50:
            self.learning_data["estimation_accuracy"] = \
                self.learning_data["estimation_accuracy"][-50:]
        
        # Update category weights if accuracy is consistently off
        category = completed_item.get("category")
        if category:
            category_accuracies = [
                entry["impact_accuracy"] for entry in self.learning_data["execution_history"][-10:]
                if entry.get("category") == category
            ]
            
            if len(category_accuracies) >= 3:
                avg_accuracy = statistics.mean(category_accuracies)
                
                # Adjust weights based on accuracy
                if avg_accuracy < 0.7:  # Consistently overestimating
                    self.learning_data["category_weights"][category] = \
                        self.weights.get(category, 0.25) * 0.9
                elif avg_accuracy > 1.3:  # Consistently underestimating  
                    self.learning_data["category_weights"][category] = \
                        self.weights.get(category, 0.25) * 1.1
        
        # Save updated learning model
        with open(self.learning_file, "w") as f:
            json.dump(self.learning_data, f, indent=2)

def main():
    """CLI entry point for enhanced value discovery."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Value Discovery Engine")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", default=".terragon/enhanced-backlog.json", 
                       help="Output file for discovered opportunities")
    
    args = parser.parse_args()
    
    discovery = EnhancedValueDiscovery(Path(args.project_root))
    opportunities = discovery.discover_value_opportunities()
    
    # Save results
    output_path = Path(args.project_root) / args.output
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "discovery_timestamp": discovery.timestamp,
            "total_opportunities": len(opportunities),
            "top_opportunity": opportunities[0] if opportunities else None,
            "opportunities": opportunities
        }, f, indent=2)
    
    print(f"Discovered {len(opportunities)} value opportunities")
    if opportunities:
        top = opportunities[0]
        print(f"Top priority: {top['title']} (score: {top['composite_score']:.1f})")

if __name__ == "__main__":
    main()