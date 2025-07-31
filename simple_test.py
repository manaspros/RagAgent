#!/usr/bin/env python3
"""
Simple test for Hybrid RAG Integration
"""

import requests
import time

def test_system():
    """Simple test of the hybrid system"""
    base_url = "http://localhost:8000"
    
    print("Testing Hybrid RAG Integration")
    print("=" * 40)
    
    # Health check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("   Health check: PASSED")
        else:
            print(f"   Health check: FAILED ({response.status_code})")
            return False
    except Exception as e:
        print(f"   Health check: ERROR - {e}")
        return False
    
    # System status
    print("\n2. Testing System Status...")
    try:
        response = requests.get(f"{base_url}/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"   System status: {status.get('status')}")
            print(f"   KG status: {status.get('kg_status')}")
            print(f"   VDB status: {status.get('vdb_status')}")
            print(f"   LLM status: {status.get('llm_status')}")
        else:
            print(f"   Status check: FAILED ({response.status_code})")
    except Exception as e:
        print(f"   Status check: ERROR - {e}")
    
    # Test query
    print("\n3. Testing Query Processing...")
    try:
        test_query = "What is the coverage for knee surgery?"
        response = requests.post(
            f"{base_url}/process_query",
            headers={"Content-Type": "application/json"},
            json={"query": test_query},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("   Query processing: PASSED")
            print(f"   Decision: {result.get('Decision')}")
            print(f"   Amount: {result.get('Amount')}")
            print(f"   Clauses found: {len(result.get('Relevant_Clauses', []))}")
        else:
            print(f"   Query processing: FAILED ({response.status_code})")
    except Exception as e:
        print(f"   Query processing: ERROR - {e}")
    
    print("\n" + "=" * 40)
    print("TEST COMPLETE")
    print("=" * 40)
    return True

if __name__ == "__main__":
    try:
        # Check if server is running
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code != 200:
                print("Server not responding correctly")
                print("Please start the server first: python hybrid_main.py")
                exit(1)
        except requests.exceptions.RequestException:
            print("Server not running")
            print("Please start the server first: python hybrid_main.py")
            exit(1)
        
        # Run tests
        test_system()
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"\nTest error: {e}")