#!/usr/bin/env python3
"""
Terragon Value Discovery Scoring Engine
Implements WSJF + ICE + Technical Debt scoring for autonomous value optimization
"""

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class WorkItem:
    id: str
    title: str
    description: str
    category: str
    files_affected: List[str]
    estimated_effort: float  # hours
    
    # WSJF Components
    user_business_value: float  # 1-10
    time_criticality: float     # 1-10
    risk_reduction: float       # 1-10
    opportunity_enablement: float # 1-10
    
    # ICE Components  
    impact: float      # 1-10
    confidence: float  # 1-10
    ease: float        # 1-10
    
    # Technical Debt
    debt_impact: float     # maintenance hours saved
    debt_interest: float   # future cost if not addressed
    hotspot_multiplier: float  # 1-5x based on churn/complexity
    
    # Security & Compliance
    is_security_issue: bool = False
    is_compliance_blocking: bool = False
    
    created_at: str = ""
    source: str = ""


class ScoringEngine:
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        try:
            import yaml
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        except (FileNotFoundError, ImportError):
            # Default weights for maturing repository
            self.config = {
                'scoring': {
                    'weights': {
                        'maturing': {
                            'wsjf': 0.6,
                            'ice': 0.1,
                            'technicalDebt': 0.2,
                            'security': 0.1
                        }
                    },
                    'thresholds': {
                        'minScore': 10,
                        'maxRisk': 0.8,
                        'securityBoost': 2.0,
                        'complianceBoost': 1.8
                    }
                }
            }
    
    def calculate_wsjf(self, item: WorkItem) -> float:
        """Calculate Weighted Shortest Job First score"""
        cost_of_delay = (
            item.user_business_value + 
            item.time_criticality + 
            item.risk_reduction + 
            item.opportunity_enablement
        )
        
        # Prevent division by zero
        job_size = max(item.estimated_effort, 0.5)
        
        return cost_of_delay / job_size
    
    def calculate_ice(self, item: WorkItem) -> float:
        """Calculate Impact * Confidence * Ease score"""
        return item.impact * item.confidence * item.ease
    
    def calculate_technical_debt_score(self, item: WorkItem) -> float:
        """Calculate technical debt value score"""
        base_debt_value = item.debt_impact + item.debt_interest
        return base_debt_value * item.hotspot_multiplier
    
    def normalize_score(self, score: float, max_expected: float = 100) -> float:
        """Normalize score to 0-1 range"""
        return min(score / max_expected, 1.0)
    
    def calculate_composite_score(self, item: WorkItem) -> Dict[str, float]:
        """Calculate the composite value score for a work item"""
        # Component scores
        wsjf = self.calculate_wsjf(item)
        ice = self.calculate_ice(item)
        tech_debt = self.calculate_technical_debt_score(item)
        
        # Get weights for repository maturity
        weights = self.config['scoring']['weights']['maturing']
        
        # Normalized component scores
        normalized_wsjf = self.normalize_score(wsjf, 40)  # Max WSJF ~40
        normalized_ice = self.normalize_score(ice, 1000)  # Max ICE = 10*10*10
        normalized_debt = self.normalize_score(tech_debt, 200)  # Max debt score
        
        # Composite base score
        composite = (
            weights['wsjf'] * normalized_wsjf +
            weights['ice'] * normalized_ice +
            weights['technicalDebt'] * normalized_debt
        )
        
        # Apply security and compliance boosts
        thresholds = self.config['scoring']['thresholds']
        if item.is_security_issue:
            composite *= thresholds['securityBoost']
        if item.is_compliance_blocking:
            composite *= thresholds['complianceBoost']
        
        # Convert to 0-100 scale
        final_score = composite * 100
        
        return {
            'wsjf': wsjf,
            'ice': ice,
            'technicalDebt': tech_debt,
            'composite': final_score,
            'normalized': {
                'wsjf': normalized_wsjf,
                'ice': normalized_ice,
                'technicalDebt': normalized_debt
            }
        }
    
    def rank_items(self, items: List[WorkItem]) -> List[Tuple[WorkItem, Dict[str, float]]]:
        """Rank work items by composite score"""
        scored_items = []
        
        for item in items:
            scores = self.calculate_composite_score(item)
            scored_items.append((item, scores))
        
        # Sort by composite score descending
        scored_items.sort(key=lambda x: x[1]['composite'], reverse=True)
        
        return scored_items
    
    def filter_executable_items(self, 
                              scored_items: List[Tuple[WorkItem, Dict[str, float]]],
                              current_work: Optional[List[str]] = None) -> List[Tuple[WorkItem, Dict[str, float]]]:
        """Filter items that can be executed based on constraints"""
        executable = []
        thresholds = self.config['scoring']['thresholds']
        
        for item, scores in scored_items:
            # Skip if score too low
            if scores['composite'] < thresholds['minScore']:
                continue
            
            # Skip if conflicts with current work
            if current_work and any(file in current_work for file in item.files_affected):
                continue
            
            # Dependency checking
            if not self.are_dependencies_met(item):
                continue
                
            # Risk assessment
            if self.assess_risk(item) > thresholds['maxRisk']:
                continue
            
            executable.append((item, scores))
        
        return executable
    
    def are_dependencies_met(self, item: WorkItem) -> bool:
        """Check if item dependencies are satisfied"""
        # Check for basic file existence for file-based items
        for file_path in item.files_affected:
            if file_path.endswith('/'):  # Directory
                continue
            if not file_path.startswith('.') and '/' in file_path:
                # Check if parent directory exists
                import os
                parent_dir = os.path.dirname(file_path)
                if parent_dir and not os.path.exists(parent_dir):
                    return False
        
        # Category-specific dependency checks
        if item.category == "security":
            # Security items can usually proceed independently
            return True
        elif item.category == "performance":
            # Performance items need existing code to optimize
            return len(item.files_affected) == 0 or any(
                os.path.exists(f) for f in item.files_affected if f.endswith(('.py', '.cpp', '.h'))
            )
        elif item.category == "documentation":
            # Documentation items need source code to document
            return True  # Can always add documentation
        elif item.category == "technical_debt":
            # Technical debt items usually work on existing code
            return True
        
        return True  # Default: assume dependencies are met
    
    def assess_risk(self, item: WorkItem) -> float:
        """Assess risk level of executing work item (0.0 = low risk, 1.0 = high risk)"""
        risk_score = 0.0
        
        # Base risk by category
        category_risk = {
            "security": 0.3,      # Medium risk - affects security posture
            "performance": 0.4,   # Medium-high risk - could impact performance
            "technical_debt": 0.2,  # Low-medium risk - usually refactoring
            "documentation": 0.1,   # Low risk - doesn't affect runtime
            "automation": 0.3,      # Medium risk - affects CI/CD
            "testing": 0.2,         # Low-medium risk - improves reliability
            "maintenance": 0.1      # Low risk - housekeeping
        }
        
        risk_score += category_risk.get(item.category, 0.3)
        
        # Risk based on effort (higher effort = higher risk)
        if item.estimated_effort > 10:
            risk_score += 0.2
        elif item.estimated_effort > 5:
            risk_score += 0.1
        
        # Risk based on files affected
        sensitive_patterns = [
            'security', 'auth', 'password', 'secret', 'key',
            'config', 'setup', 'install', 'deploy', 'production'
        ]
        
        for file_path in item.files_affected:
            file_lower = file_path.lower()
            if any(pattern in file_lower for pattern in sensitive_patterns):
                risk_score += 0.2
                break
        
        # Risk based on confidence level (lower confidence = higher risk)
        if hasattr(item, 'confidence') and item.confidence < 7:
            risk_score += 0.1
        
        # Cap risk score at 1.0
        return min(risk_score, 1.0)
    
    def generate_housekeeping_task(self) -> WorkItem:
        """Generate a housekeeping task when no high-value items exist"""
        return WorkItem(
            id=f"housekeeping-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            title="Repository maintenance",
            description="General repository housekeeping and cleanup",
            category="maintenance",
            files_affected=[],
            estimated_effort=1.0,
            user_business_value=2.0,
            time_criticality=1.0,
            risk_reduction=3.0,
            opportunity_enablement=2.0,
            impact=3.0,
            confidence=8.0,
            ease=7.0,
            debt_impact=5.0,
            debt_interest=2.0,
            hotspot_multiplier=1.0,
            source="housekeeping_generator",
            created_at=datetime.now().isoformat()
        )


def discover_work_items() -> List[WorkItem]:
    """Discover work items from various sources"""
    items = []
    
    # Example discovered items for active-inference-sim-lab
    items.extend([
        WorkItem(
            id="td-001",
            title="Add comprehensive type hints to core modules",
            description="Add complete type annotations to improve IDE support and catch type errors",
            category="technical_debt",
            files_affected=["src/python/active_inference/core.py", "src/python/active_inference/agents.py"],
            estimated_effort=4.0,
            user_business_value=3.0,
            time_criticality=2.0,
            risk_reduction=6.0,
            opportunity_enablement=5.0,
            impact=6.0,
            confidence=8.0,
            ease=7.0,
            debt_impact=15.0,
            debt_interest=8.0,
            hotspot_multiplier=2.0,
            source="static_analysis",
            created_at=datetime.now().isoformat()
        ),
        WorkItem(
            id="sec-001", 
            title="Update vulnerable dependencies",
            description="Update dependencies with known security vulnerabilities",
            category="security",
            files_affected=["requirements.txt", "pyproject.toml"],
            estimated_effort=2.0,
            user_business_value=4.0,
            time_criticality=8.0,
            risk_reduction=9.0,
            opportunity_enablement=3.0,
            impact=8.0,
            confidence=9.0,
            ease=8.0,
            debt_impact=5.0,
            debt_interest=20.0,
            hotspot_multiplier=1.0,
            is_security_issue=True,
            source="security_scan",
            created_at=datetime.now().isoformat()
        ),
        WorkItem(
            id="perf-001",
            title="Optimize free energy computation in C++ core",
            description="Profile and optimize performance bottlenecks in FreeEnergyCalculator",
            category="performance",
            files_affected=["cpp/src/core/free_energy.cpp"],
            estimated_effort=8.0,
            user_business_value=7.0,
            time_criticality=4.0,
            risk_reduction=3.0,
            opportunity_enablement=8.0,
            impact=8.0,
            confidence=6.0,
            ease=4.0,
            debt_impact=25.0,
            debt_interest=5.0,
            hotspot_multiplier=3.0,
            source="performance_profiling",
            created_at=datetime.now().isoformat()
        ),
        WorkItem(
            id="doc-001", 
            title="Add API documentation with Sphinx autodoc",
            description="Generate comprehensive API documentation from docstrings",
            category="documentation",
            files_affected=["docs/api/", "src/python/active_inference/"],
            estimated_effort=6.0,
            user_business_value=5.0,
            time_criticality=3.0,
            risk_reduction=2.0,
            opportunity_enablement=6.0,
            impact=6.0,
            confidence=8.0,
            ease=6.0,
            debt_impact=10.0,
            debt_interest=3.0,
            hotspot_multiplier=1.0,
            source="documentation_audit",
            created_at=datetime.now().isoformat()
        )
    ])
    
    return items


if __name__ == "__main__":
    # Initialize scoring engine
    engine = ScoringEngine()
    
    # Discover work items
    items = discover_work_items()
    
    # Score and rank items
    ranked_items = engine.rank_items(items)
    
    # Filter executable items
    executable = engine.filter_executable_items(ranked_items)
    
    # Display results
    print("🎯 Value-Prioritized Backlog")
    print("=" * 50)
    
    for i, (item, scores) in enumerate(executable[:10], 1):
        print(f"{i}. [{item.id.upper()}] {item.title}")
        print(f"   Composite Score: {scores['composite']:.1f}")
        print(f"   WSJF: {scores['wsjf']:.1f} | ICE: {scores['ice']:.0f} | Tech Debt: {scores['technicalDebt']:.1f}")
        print(f"   Effort: {item.estimated_effort}h | Category: {item.category}")
        if item.is_security_issue:
            print("   🔒 SECURITY ISSUE")
        print()
    
    if executable:
        next_item, next_scores = executable[0]
        print(f"🚀 Next Best Value Item: {next_item.title}")
        print(f"   Expected Value: {next_scores['composite']:.1f}")
        print(f"   Estimated Effort: {next_item.estimated_effort} hours")