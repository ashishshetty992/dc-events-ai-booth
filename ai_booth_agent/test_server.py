#!/usr/bin/env python3
"""
Simple test server to check if FastAPI is working
"""
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World", "status": "working"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("🚀 Starting test server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)