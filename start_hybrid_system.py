#!/usr/bin/env python3
"""
Hybrid RAG System Startup Script
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("Starting Enhanced Hybrid RAG System...")
    
    # Check if .env exists
    if not Path(".env").exists():
        print("Error: .env file not found. Please run setup first.")
        return
    
    # Check if hybrid_main.py exists
    if not Path("hybrid_main.py").exists():
        print("Error: hybrid_main.py not found.")
        return
    
    try:
        # Start the system
        print("Server starting at http://localhost:8000")
        print("API docs will be at http://localhost:8000/docs")
        print("Graph visualization via API endpoints")
        print("\nPress Ctrl+C to stop the server\n")
        
        subprocess.run([sys.executable, "hybrid_main.py"])
        
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()