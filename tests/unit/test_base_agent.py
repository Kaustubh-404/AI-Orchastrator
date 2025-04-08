import pytest
import asyncio
from src.agents.base_agent import BaseAgent


# Create a concrete implementation of BaseAgent for testing
class TestAgent(BaseAgent):
    def __init__(self, agent_id="test-agent", name="Test Agent", description="Agent for testing"):
        super().__init__(agent_id, name, description)
    
    def get_capabilities(self):
        return ["test_capability", "another_capability"]
    
    async def process(self, task_data, context=None):
        # Simulate processing
        if task_data.get("should_fail"):
            self.update_metrics(False, 0.5)
            return {"status": "failed", "error": "Simulated failure"}
        else:
            self.update_metrics(True, 0.5)
            return {"status": "completed", "result": "Processed: " + task_data.get("input", "")}


class TestBaseAgent:
    
    def test_initialization(self):
        agent = TestAgent()
        assert agent.agent_id == "test-agent"
        assert agent.name == "Test Agent"
        assert agent.description == "Agent for testing"
        
    def test_get_capabilities(self):
        agent = TestAgent()
        capabilities = agent.get_capabilities()
        assert isinstance(capabilities, list)
        assert "test_capability" in capabilities
        assert "another_capability" in capabilities
    
    def test_get_performance_metrics_initial(self):
        agent = TestAgent()
        metrics = agent.get_performance_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["success_rate"] == 0
        assert metrics["avg_response_time"] == 0
    
    def test_update_metrics(self):
        agent = TestAgent()
        # Update metrics with a successful request
        agent.update_metrics(True, 0.5)
        # Update metrics with a failed request
        agent.update_metrics(False, 0.3)
        
        metrics = agent.get_performance_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["success_rate"] == 50.0  # 1 out of 2 requests successful
        assert metrics["avg_response_time"] == 0.4  # (0.5 + 0.3) / 2 = 0.4
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        agent = TestAgent()
        result = await agent.health_check()
        assert result is True
        assert agent.last_health_check > 0
    
    @pytest.mark.asyncio
    async def test_process_success(self):
        agent = TestAgent()
        result = await agent.process({"input": "test data"})
        assert result["status"] == "completed"
        assert result["result"] == "Processed: test data"
        
        # Check that metrics were updated
        metrics = agent.get_performance_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["success_rate"] == 100.0
    
    @pytest.mark.asyncio
    async def test_process_failure(self):
        agent = TestAgent()
        result = await agent.process({"should_fail": True})
        assert result["status"] == "failed"
        assert "error" in result
        
        # Check that metrics were updated
        metrics = agent.get_performance_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["success_rate"] == 0.0