from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime


class RequestInput(BaseModel):
    """Model for input requests."""
    content: str = Field(..., description="The content of the request")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context for the request")


class RequestStatus(BaseModel):
    """Model for request status."""
    request_id: str = Field(..., description="Unique ID for the request")
    status: str = Field(..., description="Status of the request")
    message: Optional[str] = Field(default=None, description="Status message")
    response: Optional[str] = Field(default=None, description="Response content if available")
    error: Optional[str] = Field(default=None, description="Error message if any")
    created_at: Optional[float] = Field(default=None, description="Request creation timestamp")
    completed_at: Optional[float] = Field(default=None, description="Request completion timestamp")
    processing_time: Optional[float] = Field(default=None, description="Total processing time in seconds")


class AgentMetrics(BaseModel):
    """Model for agent performance metrics."""
    total_requests: int = Field(default=0, description="Total number of requests processed")
    success_rate: float = Field(default=0, description="Percentage of successful requests")
    avg_response_time: float = Field(default=0, description="Average response time in seconds")


class AgentInfo(BaseModel):
    """Model for agent information."""
    agent_id: str = Field(..., description="Unique ID for the agent")
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")
    capabilities: List[str] = Field(default=[], description="List of agent capabilities")
    metrics: AgentMetrics = Field(default_factory=AgentMetrics, description="Performance metrics")