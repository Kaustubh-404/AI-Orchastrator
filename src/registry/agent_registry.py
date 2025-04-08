from typing import Dict, List, Optional, Set, Any
import time
import asyncio
from ..agents.base_agent import BaseAgent


class AgentRegistry:
    """Registry for managing available AI agents and their capabilities."""
    
    def __init__(self):
        """Initialize the agent registry."""
        self.agents: Dict[str, BaseAgent] = {}
        self.capability_index: Dict[str, Set[str]] = {}
        self.health_check_interval = 60  # seconds
        self._health_check_task = None
    
    def start_health_checks(self):
        """Start periodic health checks for all agents."""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._run_health_checks())
    
    async def _run_health_checks(self):
        """Run periodic health checks on all registered agents."""
        while True:
            for agent_id, agent in self.agents.items():
                try:
                    is_healthy = await agent.health_check()
                    if not is_healthy:
                        print(f"Warning: Agent {agent_id} failed health check")
                except Exception as e:
                    print(f"Error checking health of agent {agent_id}: {str(e)}")
            
            await asyncio.sleep(self.health_check_interval)
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register a new agent with the registry.
        
        Args:
            agent: The agent to register
        """
        # Register agent
        self.agents[agent.agent_id] = agent
        
        # Index capabilities
        for capability in agent.get_capabilities():
            if capability not in self.capability_index:
                self.capability_index[capability] = set()
            self.capability_index[capability].add(agent.agent_id)
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Remove an agent from the registry.
        
        Args:
            agent_id: ID of the agent to remove
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Remove from capability index
            for capability in agent.get_capabilities():
                if capability in self.capability_index:
                    self.capability_index[capability].discard(agent_id)
                    if not self.capability_index[capability]:
                        del self.capability_index[capability]
            
            # Remove agent
            del self.agents[agent_id]
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            The agent or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_agents_for_capability(self, capability: str) -> List[BaseAgent]:
        """
        Get all agents that provide a specific capability.
        
        Args:
            capability: The capability to look for
            
        Returns:
            List of agents with the specified capability
        """
        if capability not in self.capability_index:
            return []
        
        return [self.agents[agent_id] for agent_id in self.capability_index[capability] 
                if agent_id in self.agents]
    
    def get_all_capabilities(self) -> List[str]:
        """
        Get a list of all capabilities provided by registered agents.
        
        Returns:
            List of all unique capabilities
        """
        return list(self.capability_index.keys())
    
    def get_best_agent_for_capability(self, capability: str) -> Optional[BaseAgent]:
        """
        Get the best agent for a specific capability based on performance metrics.
        
        Args:
            capability: The capability to look for
            
        Returns:
            The best agent or None if no agents provide the capability
        """
        agents = self.get_agents_for_capability(capability)
        
        if not agents:
            return None
        
        # Sort by success rate and then by response time
        sorted_agents = sorted(
            agents,
            key=lambda a: (
                a.get_performance_metrics()["success_rate"],
                -a.get_performance_metrics()["avg_response_time"]
            ),
            reverse=True
        )
        
        return sorted_agents[0]
    
    def get_all_agents(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered agents.
        
        Returns:
            List of agent information dictionaries
        """
        return [
            {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.get_capabilities(),
                "metrics": agent.get_performance_metrics(),
                "last_health_check": agent.last_health_check
            }
            for agent in self.agents.values()
        ]