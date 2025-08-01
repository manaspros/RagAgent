#!/usr/bin/env python3
"""Test the clear functions"""

import requests
import json

def test_clear_endpoints():
    """Test all clear endpoints"""
    print("=== Testing Clear Data Endpoints ===")
    
    endpoints = [
        ("Clear Vector Database", "/admin/clear-vector-db"),
        ("Clear Knowledge Graph", "/admin/clear-knowledge-graph"), 
        ("Clear All Data", "/admin/clear-all-data")
    ]
    
    for name, endpoint in endpoints:
        print(f"\nTesting {name}...")
        try:
            response = requests.post(f"http://localhost:8000{endpoint}", timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print(f"  OK {name} successful")
                    print(f"  Message: {result.get('message')}")
                else:
                    print(f"  WARNING {name} reported failure")
                    print(f"  Message: {result.get('message')}")
            else:
                print(f"  ERROR HTTP {response.status_code}")
                print(f"  Response: {response.text}")
                
        except Exception as e:
            print(f"  ERROR Exception: {e}")

def test_system_status_after_clear():
    """Check system status after clearing"""
    print("\n=== System Status After Clear ===")
    
    try:
        response = requests.get("http://localhost:8000/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            
            # Check KG status
            kg_status = status.get('kg_status', {})
            if isinstance(kg_status, dict):
                print(f"KG Status: {kg_status.get('status')}")
                print(f"KG Node Count: {kg_status.get('node_count')}")
            
            # Check VDB status
            vdb_status = status.get('vdb_status', {})
            if isinstance(vdb_status, dict):
                print(f"VDB Status: {vdb_status.get('status')}")
                print(f"VDB Document Count: {vdb_status.get('document_count')}")
                
        else:
            print(f"ERROR Failed to get status: {response.status_code}")
            
    except Exception as e:
        print(f"ERROR Status check failed: {e}")

if __name__ == "__main__":
    print("Testing Clear Data Functions\n")
    
    # First test individual endpoints (they should work now)
    test_clear_endpoints()
    
    # Then check the status
    test_system_status_after_clear()
    
    print("\n=== Test Complete ===")
    print("All clear functions should now work properly!")
    print("Check the frontend Status tab to see the Admin Controls in action.")