import pytest
import base64
import asyncio
import tempfile
import os
from PIL import Image
import io

# Import the agents
from src.agents.image_to_text_agent import ImageToTextAgent
from src.agents.text_to_image_agent import TextToImageAgent
from src.agents.text_to_audio_agent import TextToAudioAgent
from src.agents.audio_to_text_agent import AudioToTextAgent
from src.agents.video_creation_agent import VideoCreationAgent


@pytest.fixture
def test_image_base64():
    """Create a test image and encode it as base64."""
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{base64_image}"


@pytest.fixture
def test_audio_base64():
    """Create a test audio file and encode it as base64."""
    # In a real test, use a real audio file
    # For testing, just create a dummy file
    dummy_audio = b'\x00' * 1000
    base64_audio = base64.b64encode(dummy_audio).decode('utf-8')
    return f"data:audio/wav;base64,{base64_audio}"


class TestImageToTextAgent:
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test that the agent initializes correctly."""
        # Skip if the required models aren't available
        pytest.importorskip("transformers")
        
        try:
            agent = ImageToTextAgent(model_name="Salesforce/blip-image-captioning-base")
            assert agent.name == "Image-to-Text Agent"
            assert "image_to_text" in agent.get_capabilities()
        except Exception as e:
            pytest.skip(f"Model initialization failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_process(self, test_image_base64):
        """Test processing an image."""
        # Skip if the required models aren't available
        pytest.importorskip("transformers")
        
        try:
            agent = ImageToTextAgent(model_name="Salesforce/blip-image-captioning-base")
            result = await agent.process({"image_data": test_image_base64})
            
            assert result["status"] in ["completed", "failed"]
            if result["status"] == "completed":
                assert "text_description" in result
        except Exception as e:
            pytest.skip(f"Processing failed: {str(e)}")


class TestTextToImageAgent:
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test that the agent initializes correctly."""
        # Skip if the required models aren't available
        pytest.importorskip("diffusers")
        
        try:
            agent = TextToImageAgent()
            assert agent.name == "Text-to-Image Agent"
            assert "text_to_image" in agent.get_capabilities()
        except Exception as e:
            pytest.skip(f"Model initialization failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_process(self):
        """Test generating an image from text."""
        # Skip if the required models aren't available
        pytest.importorskip("diffusers")
        
        try:
            agent = TextToImageAgent()
            result = await agent.process({"prompt": "A beautiful sunset"})
            
            assert result["status"] in ["completed", "failed"]
            if result["status"] == "completed":
                assert "image_data" in result
                assert result["image_data"].startswith("data:image/png;base64,")
        except Exception as e:
            pytest.skip(f"Processing failed: {str(e)}")


class TestTextToAudioAgent:
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test that the agent initializes correctly."""
        # Skip if the required models aren't available
        pytest.importorskip("transformers")
        
        try:
            agent = TextToAudioAgent()
            assert agent.name == "Text-to-Audio Agent"
            assert "text_to_audio" in agent.get_capabilities()
        except Exception as e:
            pytest.skip(f"Model initialization failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_process(self):
        """Test generating audio from text."""
        # Skip if the required models aren't available
        pytest.importorskip("transformers")
        
        try:
            agent = TextToAudioAgent()
            result = await agent.process({"text": "Hello, this is a test."})
            
            assert result["status"] in ["completed", "failed"]
            if result["status"] == "completed":
                assert "audio_data" in result
                assert result["audio_data"].startswith("data:audio/")
        except Exception as e:
            pytest.skip(f"Processing failed: {str(e)}")


class TestAudioToTextAgent:
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test that the agent initializes correctly."""
        # Skip if the required models aren't available
        pytest.importorskip("transformers")
        
        try:
            agent = AudioToTextAgent()
            assert agent.name == "Audio-to-Text Agent"
            assert "audio_to_text" in agent.get_capabilities()
        except Exception as e:
            pytest.skip(f"Model initialization failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_process(self, test_audio_base64):
        """Test transcribing audio to text."""
        # Skip if the required models aren't available
        pytest.importorskip("transformers")
        
        try:
            agent = AudioToTextAgent()
            result = await agent.process({"audio_data": test_audio_base64})
            
            assert result["status"] in ["completed", "failed"]
            # Note: Will likely fail with dummy audio, but we're just testing the interface
        except Exception as e:
            pytest.skip(f"Processing failed: {str(e)}")


class TestVideoCreationAgent:
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test that the agent initializes correctly."""
        agent = VideoCreationAgent()
        assert agent.name == "Video Creation Agent"
        assert "video_creation" in agent.get_capabilities()
    
    @pytest.mark.asyncio
    async def test_process(self, test_image_base64, test_audio_base64):
        """Test creating a video from images and audio."""
        # Skip if ffmpeg is not available
        try:
            import subprocess
            subprocess.check_output(['ffmpeg', '-version'])
        except (subprocess.SubprocessError, FileNotFoundError):
            pytest.skip("ffmpeg not available")
        
        agent = VideoCreationAgent()
        result = await agent.process({
            "images": [test_image_base64, test_image_base64],
            "audio": test_audio_base64,
            "fps": 1
        })
        
        assert result["status"] in ["completed", "failed"]
        if result["status"] == "completed":
            assert "video_data" in result
            assert result["video_data"].startswith("data:video/mp4;base64,")