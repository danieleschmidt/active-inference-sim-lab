# Contract Testing for Active Inference API
# Ensures API compatibility across versions and services

import pytest
import json
from pathlib import Path
from typing import Dict, Any

class APIContractTester:
    """Contract testing framework for API verification."""
    
    def __init__(self, contract_dir: str = "tests/contract/schemas"):
        self.contract_dir = Path(contract_dir)
        self.contracts = self._load_contracts()
    
    def _load_contracts(self) -> Dict[str, Any]:
        """Load API contracts from schema files."""
        contracts = {}
        if self.contract_dir.exists():
            for contract_file in self.contract_dir.glob("*.json"):
                with open(contract_file) as f:
                    contracts[contract_file.stem] = json.load(f)
        return contracts
    
    def validate_request(self, endpoint: str, request_data: Dict[str, Any]) -> bool:
        """Validate request against contract schema."""
        if endpoint not in self.contracts:
            return False
            
        contract = self.contracts[endpoint]
        request_schema = contract.get("request", {})
        
        # Validate required fields
        required_fields = request_schema.get("required", [])
        for field in required_fields:
            if field not in request_data:
                return False
        
        # Validate field types
        properties = request_schema.get("properties", {})
        for field, value in request_data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if not self._validate_type(value, expected_type):
                    return False
        
        return True
    
    def validate_response(self, endpoint: str, response_data: Dict[str, Any]) -> bool:
        """Validate response against contract schema."""
        if endpoint not in self.contracts:
            return False
            
        contract = self.contracts[endpoint]
        response_schema = contract.get("response", {})
        
        # Validate required fields
        required_fields = response_schema.get("required", [])
        for field in required_fields:
            if field not in response_data:
                return False
        
        # Validate field types
        properties = response_schema.get("properties", {})
        for field, value in response_data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if not self._validate_type(value, expected_type):
                    return False
        
        return True
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type against expected type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, assume valid

@pytest.fixture
def contract_tester():
    """Provide contract testing instance."""
    return APIContractTester()

class TestActiveInferenceAPIContracts:
    """Test API contracts for Active Inference endpoints."""
    
    def test_agent_creation_contract(self, contract_tester):
        """Test agent creation API contract."""
        # Valid request
        valid_request = {
            "state_dim": 4,
            "obs_dim": 8,
            "action_dim": 2,
            "inference_method": "variational"
        }
        
        assert contract_tester.validate_request("create_agent", valid_request)
        
        # Valid response
        valid_response = {
            "agent_id": "agent_123",
            "status": "created",
            "config": {
                "state_dim": 4,
                "obs_dim": 8,
                "action_dim": 2
            }
        }
        
        assert contract_tester.validate_response("create_agent", valid_response)
    
    def test_inference_contract(self, contract_tester):
        """Test inference API contract."""
        valid_request = {
            "agent_id": "agent_123",
            "observations": [1.0, 2.0, 3.0, 4.0],
            "prior_beliefs": {
                "mean": [0.0, 0.0],
                "variance": [1.0, 1.0]
            }
        }
        
        assert contract_tester.validate_request("infer_states", valid_request)
        
        valid_response = {
            "beliefs": {
                "mean": [0.5, 0.3],
                "variance": [0.8, 0.9]
            },
            "free_energy": 2.34,
            "inference_time_ms": 15.2
        }
        
        assert contract_tester.validate_response("infer_states", valid_response)
    
    def test_planning_contract(self, contract_tester):
        """Test action planning API contract."""
        valid_request = {
            "agent_id": "agent_123", 
            "current_beliefs": {
                "mean": [0.5, 0.3],
                "variance": [0.8, 0.9]
            },
            "horizon": 5,
            "objective": "expected_free_energy"
        }
        
        assert contract_tester.validate_request("plan_action", valid_request)
        
        valid_response = {
            "action": [0.2, -0.1],
            "expected_free_energy": 1.87,
            "planning_time_ms": 23.1,
            "trajectory": [
                {"step": 0, "action": [0.2, -0.1], "efe": 1.87},
                {"step": 1, "action": [0.1, 0.0], "efe": 1.45}
            ]
        }
        
        assert contract_tester.validate_response("plan_action", valid_response)