"""
Integration tests for Auto-Scaling System
Generation 4: Quality Gates - Comprehensive Integration Testing
"""

import pytest
import time
import threading
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src" / "python"))

from active_inference.scalability.auto_scaling import (
    AutoScaler, InstanceManager, MetricsCollector, 
    ScalingRule, ScalingTrigger, ScalingDirection, ResourceMetrics,
    ScalingEvent, create_agent_factory, auto_scaled_load_test
)


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or f"mock_agent_{int(time.time() * 1000)}"
        self.created_at = time.time()
        self.requests_handled = 0
        self.processing_times = []
        self.is_shutdown = False
        
    def process_request(self, data: Any) -> Any:
        """Simulate processing a request."""
        start_time = time.time()
        time.sleep(0.01)  # Simulate processing time
        processing_time = time.time() - start_time
        
        self.requests_handled += 1
        self.processing_times.append(processing_time)
        
        return f"processed_{data}_by_{self.agent_id}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'agent_id': self.agent_id,
            'requests_handled': self.requests_handled,
            'avg_processing_time': sum(self.processing_times) / max(1, len(self.processing_times)),
            'created_at': self.created_at
        }
    
    def shutdown(self):
        """Shutdown the agent."""
        self.is_shutdown = True


class TestInstanceManager:
    """Test InstanceManager functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.agent_factory = lambda: MockAgent()
        self.instance_manager = InstanceManager(self.agent_factory)
    
    def test_create_instance(self):
        """Test creating new agent instances."""
        instance_id = self.instance_manager.create_instance()
        
        assert instance_id is not None
        assert instance_id.startswith("instance_")
        assert instance_id in self.instance_manager.instances
        
        instance = self.instance_manager.get_instance(instance_id)
        assert isinstance(instance, MockAgent)
        assert instance.agent_id is not None
    
    def test_terminate_instance(self):
        """Test terminating agent instances."""
        instance_id = self.instance_manager.create_instance()
        
        # Verify instance exists
        assert self.instance_manager.get_instance(instance_id) is not None
        
        # Terminate instance
        success = self.instance_manager.terminate_instance(instance_id)
        
        assert success is True
        assert self.instance_manager.get_instance(instance_id) is None
        assert instance_id not in self.instance_manager.instances
    
    def test_terminate_nonexistent_instance(self):
        """Test terminating non-existent instance."""
        success = self.instance_manager.terminate_instance("nonexistent")
        assert success is False
    
    def test_instance_status_management(self):
        """Test instance status tracking."""
        instance_id = self.instance_manager.create_instance()
        
        # Initially idle
        idle_instances = self.instance_manager.get_idle_instances()
        assert instance_id in idle_instances
        
        # Mark as busy
        self.instance_manager.mark_instance_busy(instance_id)
        idle_instances = self.instance_manager.get_idle_instances()
        assert instance_id not in idle_instances
        
        # Mark as idle again
        self.instance_manager.mark_instance_idle(instance_id)
        idle_instances = self.instance_manager.get_idle_instances()
        assert instance_id in idle_instances
    
    def test_instance_statistics_update(self):
        """Test updating instance statistics."""
        instance_id = self.instance_manager.create_instance()
        
        # Update stats
        self.instance_manager.update_instance_stats(instance_id, 0.5)
        self.instance_manager.update_instance_stats(instance_id, 0.3)
        
        stats = self.instance_manager.get_instance_statistics()
        
        assert stats['active_instances'] == 1
        assert stats['total_instances_created'] == 1
        assert stats['total_requests_handled'] == 2
        assert stats['avg_processing_time'] == 0.4  # (0.5 + 0.3) / 2
    
    def test_multiple_instances(self):
        """Test managing multiple instances."""
        instance_ids = []
        
        # Create multiple instances
        for _ in range(5):
            instance_id = self.instance_manager.create_instance()
            instance_ids.append(instance_id)
        
        assert self.instance_manager.get_instance_count() == 5
        
        # Mark some as busy
        self.instance_manager.mark_instance_busy(instance_ids[0])
        self.instance_manager.mark_instance_busy(instance_ids[1])
        
        stats = self.instance_manager.get_instance_statistics()
        assert stats['active_instances'] == 5
        assert stats['idle_instances'] == 3
        assert stats['busy_instances'] == 2
        
        # Terminate some instances
        for instance_id in instance_ids[:2]:
            self.instance_manager.terminate_instance(instance_id)
        
        assert self.instance_manager.get_instance_count() == 3


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.metrics_collector = MetricsCollector(collection_interval=0.1)
    
    def teardown_method(self):
        """Cleanup after each test."""
        self.metrics_collector.stop_collection()
    
    def test_metrics_collection(self):
        """Test basic metrics collection."""
        self.metrics_collector.start_collection()
        
        # Wait for some metrics to be collected
        time.sleep(0.3)
        
        current_metrics = self.metrics_collector.get_current_metrics()
        
        assert current_metrics.timestamp > 0
        assert current_metrics.cpu_percent >= 0
        assert current_metrics.memory_percent >= 0
        assert current_metrics.memory_available_gb >= 0
    
    def test_request_recording(self):
        """Test recording requests for metrics."""
        # Record some requests
        self.metrics_collector.record_request(0.1, False)
        self.metrics_collector.record_request(0.2, False)
        self.metrics_collector.record_request(0.3, True)  # Error
        
        time.sleep(0.1)  # Wait for next collection cycle
        
        current_metrics = self.metrics_collector.get_current_metrics()
        
        assert current_metrics.requests_per_second > 0
        assert current_metrics.error_rate > 0
        assert current_metrics.avg_response_time > 0
    
    def test_custom_metrics(self):
        """Test custom metric providers."""
        def custom_metric_1():
            return 42.5
        
        def custom_metric_2():
            return 100.0
        
        self.metrics_collector.register_custom_metric("custom_1", custom_metric_1)
        self.metrics_collector.register_custom_metric("custom_2", custom_metric_2)
        
        self.metrics_collector.start_collection()
        time.sleep(0.2)
        
        current_metrics = self.metrics_collector.get_current_metrics()
        
        assert "custom_1" in current_metrics.custom_metrics
        assert "custom_2" in current_metrics.custom_metrics
        assert current_metrics.custom_metrics["custom_1"] == 42.5
        assert current_metrics.custom_metrics["custom_2"] == 100.0
    
    def test_metrics_history(self):
        """Test metrics history tracking."""
        self.metrics_collector.start_collection()
        
        # Record some requests
        for i in range(10):
            self.metrics_collector.record_request(0.1 * i, i % 3 == 0)
            time.sleep(0.05)
        
        time.sleep(0.2)
        
        # Get history
        cpu_history = self.metrics_collector.get_metric_history("cpu_percent", 1.0)
        response_time_history = self.metrics_collector.get_metric_history("avg_response_time", 1.0)
        
        assert len(cpu_history) > 0
        assert len(response_time_history) > 0
        
        # Test average calculation
        avg_cpu = self.metrics_collector.get_average_metric("cpu_percent", 1.0)
        assert avg_cpu >= 0
    
    def test_queue_depth_update(self):
        """Test queue depth updates."""
        self.metrics_collector.update_queue_depth(15)
        
        current_metrics = self.metrics_collector.get_current_metrics()
        assert current_metrics.queue_depth == 15
    
    def test_active_instances_update(self):
        """Test active instances count update."""
        self.metrics_collector.update_active_instances(7)
        
        current_metrics = self.metrics_collector.get_current_metrics()
        assert current_metrics.active_instances == 7


class TestAutoScaler:
    """Test AutoScaler functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.agent_factory = create_agent_factory(MockAgent)
        self.instance_manager = InstanceManager(self.agent_factory)
        self.metrics_collector = MetricsCollector(collection_interval=0.1)
        
        # Custom scaling rules for testing
        self.scaling_rules = [
            ScalingRule(
                name="test_cpu_scaling",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                scale_up_threshold=75.0,
                scale_down_threshold=25.0,
                min_instances=1,
                max_instances=5,
                cooldown_period=0.2,  # Short cooldown for testing
                evaluation_window=0.3
            ),
            ScalingRule(
                name="test_response_time_scaling",
                trigger=ScalingTrigger.RESPONSE_TIME,
                scale_up_threshold=1.0,
                scale_down_threshold=0.1,
                min_instances=1,
                max_instances=8,
                cooldown_period=0.2,
                evaluation_window=0.3,
                priority=2
            )
        ]
        
        self.auto_scaler = AutoScaler(
            self.instance_manager,
            self.metrics_collector,
            self.scaling_rules,
            decision_interval=0.2
        )
    
    def teardown_method(self):
        """Cleanup after each test."""
        self.auto_scaler.stop_auto_scaling()
        self.metrics_collector.stop_collection()
    
    def test_initial_state(self):
        """Test auto-scaler initial state."""
        assert len(self.auto_scaler.scaling_rules) == 2
        assert self.auto_scaler.total_scale_ups == 0
        assert self.auto_scaler.total_scale_downs == 0
        assert self.auto_scaler.total_scaling_decisions == 0
    
    def test_scaling_rule_validation(self):
        """Test scaling rule validation."""
        # Invalid rule: min_instances < 1
        with pytest.raises(ValueError, match="min_instances must be >= 1"):
            invalid_rule = ScalingRule(
                name="invalid",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                scale_up_threshold=80.0,
                scale_down_threshold=20.0,
                min_instances=0
            )
            AutoScaler(self.instance_manager, self.metrics_collector, [invalid_rule])
        
        # Invalid rule: max_instances < min_instances
        with pytest.raises(ValueError, match="max_instances must be >= min_instances"):
            invalid_rule = ScalingRule(
                name="invalid2",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                scale_up_threshold=80.0,
                scale_down_threshold=20.0,
                min_instances=5,
                max_instances=3
            )
            AutoScaler(self.instance_manager, self.metrics_collector, [invalid_rule])
        
        # Invalid rule: scale_up_threshold <= scale_down_threshold
        with pytest.raises(ValueError, match="scale_up_threshold must be > scale_down_threshold"):
            invalid_rule = ScalingRule(
                name="invalid3",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                scale_up_threshold=20.0,
                scale_down_threshold=80.0
            )
            AutoScaler(self.instance_manager, self.metrics_collector, [invalid_rule])
    
    def test_manual_scaling(self):
        """Test manual scaling operations."""
        # Initial state should have 0 instances
        assert self.instance_manager.get_instance_count() == 0
        
        # Scale up manually
        success = self.auto_scaler.manual_scale(3, "manual test")
        assert success is True
        assert self.instance_manager.get_instance_count() == 3
        
        # Scale down manually
        success = self.auto_scaler.manual_scale(1, "scale down test")
        assert success is True
        assert self.instance_manager.get_instance_count() == 1
        
        # Check scaling events
        events = self.auto_scaler.get_recent_scaling_events()
        assert len(events) == 2
        assert events[0]['rule_name'] == "manual_scaling"  # Most recent first
        assert events[0]['direction'] == "down"
        assert events[1]['direction'] == "up"
    
    @patch('psutil.cpu_percent')
    def test_cpu_based_scaling(self, mock_cpu):
        """Test CPU-based auto-scaling."""
        # Start with one instance
        self.auto_scaler.manual_scale(1, "initial")
        
        # Mock high CPU usage
        mock_cpu.return_value = 85.0
        
        # Start auto-scaling
        self.metrics_collector.start_collection()
        self.auto_scaler.start_auto_scaling()
        
        # Wait for scaling decisions
        time.sleep(0.8)
        
        # Should have scaled up due to high CPU
        final_count = self.instance_manager.get_instance_count()
        assert final_count > 1
        
        stats = self.auto_scaler.get_scaling_statistics()
        assert stats['total_scale_ups'] > 0
        
        # Now mock low CPU usage
        mock_cpu.return_value = 15.0
        
        # Wait for scale down decision
        time.sleep(0.8)
        
        # Should have scaled down
        new_count = self.instance_manager.get_instance_count()
        assert new_count < final_count
        
        final_stats = self.auto_scaler.get_scaling_statistics()
        assert final_stats['total_scale_downs'] > 0
    
    def test_response_time_based_scaling(self):
        """Test response time-based auto-scaling."""
        # Start with one instance
        self.auto_scaler.manual_scale(1, "initial")
        
        # Simulate high response times
        for _ in range(10):
            self.metrics_collector.record_request(2.0, False)  # High response time
            time.sleep(0.02)
        
        self.metrics_collector.start_collection()
        self.auto_scaler.start_auto_scaling()
        
        # Wait for scaling decision
        time.sleep(0.8)
        
        # Should have scaled up due to high response times
        assert self.instance_manager.get_instance_count() > 1
        
        stats = self.auto_scaler.get_scaling_statistics()
        assert stats['total_scale_ups'] > 0
    
    def test_scaling_cooldown(self):
        """Test scaling cooldown mechanism."""
        # Start with one instance
        self.auto_scaler.manual_scale(1, "initial")
        
        # Create a rule with a longer cooldown for testing
        test_rule = ScalingRule(
            name="cooldown_test",
            trigger=ScalingTrigger.CPU_UTILIZATION,
            scale_up_threshold=50.0,
            scale_down_threshold=10.0,
            cooldown_period=2.0  # 2 second cooldown
        )
        
        self.auto_scaler.add_scaling_rule(test_rule)
        
        # Force a scaling event
        with patch('psutil.cpu_percent', return_value=80.0):
            self.metrics_collector.start_collection()
            self.auto_scaler.start_auto_scaling()
            
            time.sleep(0.5)
            
            first_count = self.instance_manager.get_instance_count()
            
            # Try to trigger another scaling event immediately
            time.sleep(0.3)
            
            second_count = self.instance_manager.get_instance_count()
            
            # Should not have scaled again due to cooldown
            assert second_count == first_count
    
    def test_rule_management(self):
        """Test adding, removing, and managing scaling rules."""
        initial_rule_count = len(self.auto_scaler.scaling_rules)
        
        # Add a new rule
        new_rule = ScalingRule(
            name="test_new_rule",
            trigger=ScalingTrigger.QUEUE_DEPTH,
            scale_up_threshold=20.0,
            scale_down_threshold=5.0
        )
        
        self.auto_scaler.add_scaling_rule(new_rule)
        assert len(self.auto_scaler.scaling_rules) == initial_rule_count + 1
        
        # Disable a rule
        success = self.auto_scaler.disable_rule("test_new_rule")
        assert success is True
        
        # Find the rule and verify it's disabled
        disabled_rule = next(r for r in self.auto_scaler.scaling_rules if r.name == "test_new_rule")
        assert disabled_rule.enabled is False
        
        # Enable the rule again
        success = self.auto_scaler.enable_rule("test_new_rule")
        assert success is True
        assert disabled_rule.enabled is True
        
        # Remove the rule
        success = self.auto_scaler.remove_scaling_rule("test_new_rule")
        assert success is True
        assert len(self.auto_scaler.scaling_rules) == initial_rule_count
        
        # Try to remove non-existent rule
        success = self.auto_scaler.remove_scaling_rule("nonexistent")
        assert success is False
    
    def test_scaling_statistics(self):
        """Test scaling statistics collection."""
        # Perform some manual scaling operations
        self.auto_scaler.manual_scale(3, "test")
        self.auto_scaler.manual_scale(1, "test")
        
        stats = self.auto_scaler.get_scaling_statistics()
        
        # Verify basic statistics
        assert stats['current_instances'] == 1
        assert stats['total_scale_ups'] >= 1
        assert stats['total_scale_downs'] >= 1
        assert 'rule_configurations' in stats
        assert len(stats['rule_configurations']) == 2
        
        # Verify rule configurations
        rule_config = stats['rule_configurations'][0]
        assert 'name' in rule_config
        assert 'trigger' in rule_config
        assert 'enabled' in rule_config
        assert 'scale_up_threshold' in rule_config
        assert 'scale_down_threshold' in rule_config
    
    def test_recent_scaling_events(self):
        """Test recent scaling events retrieval."""
        # Perform scaling operations
        self.auto_scaler.manual_scale(2, "test up")
        time.sleep(0.1)
        self.auto_scaler.manual_scale(1, "test down")
        
        # Get recent events
        events = self.auto_scaler.get_recent_scaling_events(1)  # Last 1 hour
        
        assert len(events) >= 2
        
        # Events should be sorted by timestamp (newest first)
        assert events[0]['timestamp'] > events[1]['timestamp']
        
        # Verify event structure
        event = events[0]
        assert 'rule_name' in event
        assert 'direction' in event
        assert 'old_count' in event
        assert 'new_count' in event
        assert 'reason' in event
        assert 'success' in event
        assert 'duration' in event
    
    def test_predictive_scaling(self):
        """Test predictive scaling functionality."""
        # Create auto-scaler with predictive scaling enabled
        predictive_scaler = AutoScaler(
            self.instance_manager,
            self.metrics_collector,
            self.scaling_rules,
            decision_interval=0.1,
            enable_predictive_scaling=True
        )
        
        # Start with one instance
        predictive_scaler.manual_scale(1, "initial")
        
        # Generate trending metrics (increasing CPU usage)
        with patch('psutil.cpu_percent') as mock_cpu:
            cpu_values = [50, 55, 60, 65, 70, 75]  # Increasing trend
            mock_cpu.side_effect = cpu_values
            
            self.metrics_collector.start_collection()
            predictive_scaler.start_auto_scaling()
            
            # Wait for enough data points and scaling decisions
            time.sleep(1.0)
            
            # Should have triggered scaling due to trend
            final_count = self.instance_manager.get_instance_count()
            stats = predictive_scaler.get_scaling_statistics()
            
            # Verify predictive scaling had some effect
            assert stats['total_scaling_decisions'] > 0
        
        predictive_scaler.stop_auto_scaling()


class TestAgentFactory:
    """Test agent factory functionality."""
    
    def test_simple_factory(self):
        """Test simple agent factory creation."""
        factory = create_agent_factory(MockAgent)
        
        agent = factory()
        assert isinstance(agent, MockAgent)
        assert agent.agent_id is not None
    
    def test_factory_with_args(self):
        """Test agent factory with initialization arguments."""
        factory = create_agent_factory(MockAgent, "test_agent_123")
        
        agent = factory()
        assert isinstance(agent, MockAgent)
        assert agent.agent_id == "test_agent_123"
    
    def test_factory_with_kwargs(self):
        """Test agent factory with keyword arguments."""
        factory = create_agent_factory(MockAgent, agent_id="kwargs_agent")
        
        agent = factory()
        assert isinstance(agent, MockAgent)
        assert agent.agent_id == "kwargs_agent"


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_load_test_scenario(self):
        """Test complete load test scenario with auto-scaling."""
        # Setup components
        agent_factory = create_agent_factory(MockAgent)
        instance_manager = InstanceManager(agent_factory)
        metrics_collector = MetricsCollector(collection_interval=0.05)
        
        scaling_rules = [
            ScalingRule(
                name="load_test_cpu",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                scale_up_threshold=60.0,
                scale_down_threshold=20.0,
                min_instances=1,
                max_instances=5,
                cooldown_period=0.1,
                evaluation_window=0.2
            )
        ]
        
        auto_scaler = AutoScaler(
            instance_manager,
            metrics_collector,
            scaling_rules,
            decision_interval=0.1
        )
        
        try:
            # Simulate load test with auto-scaling
            with patch('psutil.cpu_percent') as mock_cpu:
                # Phase 1: Low load
                mock_cpu.return_value = 30.0
                
                async with auto_scaled_load_test(auto_scaler, duration_minutes=0.05, initial_instances=2):
                    # Verify initial setup
                    assert instance_manager.get_instance_count() == 2
                    
                    # Wait a bit
                    await asyncio.sleep(0.2)
                    
                    # Phase 2: High load
                    mock_cpu.return_value = 80.0
                    
                    # Simulate request load
                    for _ in range(20):
                        metrics_collector.record_request(0.1, False)
                        await asyncio.sleep(0.01)
                    
                    # Wait for scaling response
                    await asyncio.sleep(0.3)
                    
                    # Should have scaled up
                    scaled_up_count = instance_manager.get_instance_count()
                    assert scaled_up_count > 2
                    
                    # Phase 3: Load decreases
                    mock_cpu.return_value = 15.0
                    
                    await asyncio.sleep(0.3)
            
            # Verify final statistics
            final_stats = auto_scaler.get_scaling_statistics()
            assert final_stats['total_scale_ups'] > 0
            
            events = auto_scaler.get_recent_scaling_events()
            assert len(events) > 0
            
        finally:
            auto_scaler.stop_auto_scaling()
            metrics_collector.stop_collection()
    
    def test_multi_rule_scaling_priority(self):
        """Test scaling with multiple rules and priority handling."""
        # Setup components
        agent_factory = create_agent_factory(MockAgent)
        instance_manager = InstanceManager(agent_factory)
        metrics_collector = MetricsCollector(collection_interval=0.05)
        
        # Rules with different priorities
        scaling_rules = [
            ScalingRule(
                name="high_priority_response_time",
                trigger=ScalingTrigger.RESPONSE_TIME,
                scale_up_threshold=0.5,
                scale_down_threshold=0.1,
                min_instances=1,
                max_instances=3,
                priority=3,  # Highest priority
                cooldown_period=0.1
            ),
            ScalingRule(
                name="medium_priority_cpu",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                scale_up_threshold=70.0,
                scale_down_threshold=30.0,
                min_instances=1,
                max_instances=4,
                priority=2,
                cooldown_period=0.1
            ),
            ScalingRule(
                name="low_priority_memory",
                trigger=ScalingTrigger.MEMORY_UTILIZATION,
                scale_up_threshold=80.0,
                scale_down_threshold=40.0,
                min_instances=1,
                max_instances=2,
                priority=1,  # Lowest priority
                cooldown_period=0.1
            )
        ]
        
        auto_scaler = AutoScaler(
            instance_manager,
            metrics_collector,
            scaling_rules,
            decision_interval=0.1
        )
        
        try:
            # Start with one instance
            auto_scaler.manual_scale(1, "initial")
            
            # Simulate conditions that trigger multiple rules
            with patch('psutil.cpu_percent', return_value=75.0), \
                 patch('psutil.virtual_memory') as mock_memory:
                
                # Mock high memory usage
                mock_memory.return_value.percent = 85.0
                
                # Simulate high response times (highest priority)
                for _ in range(10):
                    metrics_collector.record_request(1.0, False)  # High response time
                    time.sleep(0.01)
                
                metrics_collector.start_collection()
                auto_scaler.start_auto_scaling()
                
                # Wait for scaling decision
                time.sleep(0.5)
                
                # Should have scaled based on highest priority rule
                events = auto_scaler.get_recent_scaling_events()
                
                # Find the scaling event (excluding manual scaling)
                auto_scaling_events = [e for e in events if e['rule_name'] != 'manual_scaling']
                
                if auto_scaling_events:
                    # Should have applied the highest priority rule
                    assert auto_scaling_events[0]['rule_name'] == "high_priority_response_time"
        
        finally:
            auto_scaler.stop_auto_scaling()
            metrics_collector.stop_collection()
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery in scaling operations."""
        # Create a faulty agent factory that fails sometimes
        failure_count = 0
        
        def faulty_agent_factory():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:  # First two attempts fail
                raise RuntimeError(f"Factory failure #{failure_count}")
            return MockAgent(f"recovery_agent_{failure_count}")
        
        instance_manager = InstanceManager(faulty_agent_factory)
        metrics_collector = MetricsCollector(collection_interval=0.1)
        
        scaling_rules = [
            ScalingRule(
                name="error_test_rule",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                scale_up_threshold=50.0,
                scale_down_threshold=20.0,
                min_instances=1,
                max_instances=5,
                cooldown_period=0.1
            )
        ]
        
        auto_scaler = AutoScaler(
            instance_manager,
            metrics_collector,
            scaling_rules,
            decision_interval=0.1
        )
        
        try:
            # Try to create instances - first two should fail
            with pytest.raises(RuntimeError, match="Factory failure #1"):
                instance_manager.create_instance()
            
            with pytest.raises(RuntimeError, match="Factory failure #2"):
                instance_manager.create_instance()
            
            # Third attempt should succeed
            instance_id = instance_manager.create_instance()
            assert instance_id is not None
            
            instance = instance_manager.get_instance(instance_id)
            assert instance.agent_id == "recovery_agent_3"
            
            # Test manual scaling with partial failures
            success = auto_scaler.manual_scale(3, "error recovery test")
            
            # Should have created as many instances as possible
            final_count = instance_manager.get_instance_count()
            assert final_count >= 1  # At least the original instance
            
            # Check scaling events for error information
            events = auto_scaler.get_recent_scaling_events()
            scaling_event = next((e for e in events if e['rule_name'] == 'manual_scaling'), None)
            assert scaling_event is not None
            
        finally:
            auto_scaler.stop_auto_scaling()
            metrics_collector.stop_collection()


if __name__ == "__main__":
    pytest.main([__file__])