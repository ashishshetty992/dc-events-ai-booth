#!/usr/bin/env python3
"""
Railway entry point - imports the FastAPI app from ai_booth_agent
"""

import sys
import os

# Add ai_booth_agent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_booth_agent'))

# Import the FastAPI app
from ai_booth_agent.main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
