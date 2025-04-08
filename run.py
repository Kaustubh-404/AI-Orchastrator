import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if GROQ_API_KEY is set
if not os.environ.get("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY environment variable is not set.")
    print("Please set it with your Groq API key or create a .env file with GROQ_API_KEY=your_api_key")
    sys.exit(1)

# Import and run the FastAPI app
from src.api.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))