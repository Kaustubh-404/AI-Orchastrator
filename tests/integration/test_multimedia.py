import pytest
from fastapi.testclient import TestClient
import base64
import os
import tempfile
from PIL import Image
import io

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def test_image():
    """Create a test image."""
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return {
        'content': img_byte_arr.getvalue(),
        'filename': 'test_image.png',
        'content_type': 'image/png'
    }


@pytest.fixture
def test_audio():
    """Create a test audio file (just a placeholder for testing)."""
    # In a real test, you would use a real audio file
    # For testing, we just create a small file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(b'\x00' * 1000)  # Dummy content
        audio_path = f.name
    
    with open(audio_path, 'rb') as f:
        content = f.read()
    
    os.unlink(audio_path)  # Clean up
    
    return {
        'content': content,
        'filename': 'test_audio.wav',
        'content_type': 'audio/wav'
    }


def test_upload_image(client, test_image):
    """Test uploading an image file."""
    files = {'file': (test_image['filename'], test_image['content'], test_image['content_type'])}
    response = client.post("/api/v1/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert "data_uri" in data
    assert data["data_uri"].startswith("data:image/png;base64,")


def test_create_request_with_media(client, test_image):
    """Test creating a request with media."""
    files = {'files': (test_image['filename'], test_image['content'], test_image['content_type'])}
    data = {'content': "Describe this image"}
    
    response = client.post("/api/v1/requests-with-media", files=files, data=data)
    
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert data["status"] == "processing"


def test_text_to_image_workflow(client):
    """Test the text-to-image workflow."""
    # Create a request for image generation
    response = client.post(
        "/api/v1/requests",
        json={"content": "Generate an image of a sunset over mountains"}
    )
    assert response.status_code == 200
    request_id = response.json()["request_id"]
    
    # In a real test, we would wait for processing to complete
    # For this test, we just check that the endpoint exists
    response = client.get(f"/api/v1/requests/{request_id}")
    assert response.status_code == 200
    
    # Check that the media endpoint exists
    # Note: This will likely return 404 in tests since the request isn't actually processed
    response = client.get(f"/api/v1/media/{request_id}")
    assert response.status_code in [404, 200, 400]  # 404 if not found, 400 if not completed


def test_motivational_reel_workflow(client):
    """Test the full motivational reel creation workflow."""
    # Create a request for a motivational reel
    response = client.post(
        "/api/v1/requests",
        json={
            "content": "Create a 5-second motivational reel about exercise with energetic background"
        }
    )
    assert response.status_code == 200
    request_id = response.json()["request_id"]
    
    # Check the request status
    response = client.get(f"/api/v1/requests/{request_id}")
    assert response.status_code == 200
    
    # Note: In a real test with longer timeouts, we would poll until completion
    # and then verify the video was created successfully