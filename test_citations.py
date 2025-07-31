#!/usr/bin/env python3
"""
Test the citation system with a direct API call
"""

import requests
import json

def test_citation_system():
    """Test the enhanced citation system"""
    
    # Test query
    query = "Is knee surgery covered for a 45-year-old in Mumbai?"
    
    # Make API call
    response = requests.post(
        "http://localhost:8000/process_query",
        headers={"Content-Type": "application/json"},
        json={"query": query},
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print("=== ENHANCED CITATION SYSTEM TEST ===")
        print(f"Query: {query}")
        print(f"Decision: {result.get('Decision')}")
        print(f"Amount: {result.get('Amount')}")
        print(f"\nJustification:")
        print(result.get('Justification'))
        
        print(f"\nRelevant Clauses with Citations:")
        for i, clause in enumerate(result.get('Relevant_Clauses', [])):
            print(f"\n{i+1}. Source: {clause.get('retrieval_source')}")
            print(f"   Document: {clause.get('document_id')}")
            print(f"   Section: {clause.get('page_section')}")
            print(f"   Text: {clause.get('clause_text', '')[:200]}...")
        
        print(f"\nProcessing Info:")
        proc_info = result.get('Processing_Info', {})
        for key, value in proc_info.items():
            print(f"  {key}: {value}")
    
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_citation_system()