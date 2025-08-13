"""
Vercel API entry point for the AI Booth Agent backend.
This file serves as the serverless function handler for Vercel.
"""

import sys
import os

# Add the ai_booth_agent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai_booth_agent'))

# Import the FastAPI app
from main import app
from fastapi.middleware.cors import CORSMiddleware

# Configure CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "environment": "vercel"}

# Vercel handler
handler = app
