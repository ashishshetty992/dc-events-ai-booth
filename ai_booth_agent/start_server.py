#!/usr/bin/env python3
"""
Server startup script with error handling
"""
import sys
import traceback

def start_server():
    try:
        print("🔍 Step 1: Importing modules...")
        from main import app
        print("✅ FastAPI app imported successfully")
        
        print("🔍 Step 2: Initializing database...")
        from db import init_db
        init_db()
        print("✅ Database initialized successfully")
        
        print("🔍 Step 3: Starting uvicorn server...")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    start_server()