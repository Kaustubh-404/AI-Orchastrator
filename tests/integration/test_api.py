import pytest
from fastapi.testclient import TestClient
import json
import time

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_request(client):
    """Test creating a new request."""
    response = client.post(
        "/api/v1/requests",
        json={"content": "Test request for integration test"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert data["status"] == "processing"


def test_get_request_status(client):
    """Test getting the status of a request."""
    # First create a request
    create_response = client.post(
        "/api/v1/requests",
        json={"content": "Another test request"}
    )
    assert create_response.status_code == 200
    request_id = create_response.json()["request_id"]
    
    # Then get its status
    status_response = client.get(f"/api/v1/requests/{request_id}")
    assert status_response.status_code == 200
    data = status_response.json()
    assert data["request_id"] == request_id
    # Status could be processing or completed, depending on timing
    assert data["status"] in ["processing", "completed", "failed"]


def test_list_requests(client):
    """Test listing all requests."""
    response = client.get("/api/v1/requests")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_list_agents(client):
    """Test listing all agents."""
    response = client.get("/api/v1/agents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # At least one agent should be registered (the text generation agent)
    assert len(data) >= 1