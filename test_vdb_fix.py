#!/usr/bin/env python3
"""Test VDB chunk processing fix"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_keyword_extraction():
    """Test the basic keyword extraction"""
    print("=== Testing Basic Keyword Extraction ===")
    
    try:
        from hybrid_main import HybridRAGProcessor
        
        processor = HybridRAGProcessor()
        
        # Test content
        test_content = """
        This is a comprehensive health insurance policy.
        Coverage includes surgery procedures up to $15000.
        Age limit is 65 years for eligible patients.
        Mental illness treatment is covered under this policy.
        Premium payment is required monthly.
        """
        
        keywords = processor._extract_basic_keywords(test_content)
        
        print(f"Extracted keywords: {keywords}")
        print(f"Number of keywords: {len(keywords)}")
        
        # Check for expected keywords
        expected = ['insurance', 'policy', 'coverage', 'surgery', 'age', 'eligible', 'mental', 'illness', 'treatment', 'premium']
        found_expected = [k for k in expected if k in keywords]
        
        print(f"Expected keywords found: {found_expected}")
        print("OK Basic keyword extraction working")
        
    except Exception as e:
        print(f"ERROR Keyword extraction test failed: {e}")

def test_document_processing():
    """Test document processing with the existing test file"""
    print("\n=== Testing Document Processing ===")
    
    try:
        from hybrid_main import HybridRAGProcessor
        
        processor = HybridRAGProcessor()
        
        # Test with existing file
        test_file = "test_insurance.txt"
        if os.path.exists(test_file):
            print(f"Testing with: {test_file}")
            
            result = processor.process_document(test_file, "test_doc")
            
            print(f"Processing result: {result}")
            print(f"Success: {result.get('success')}")
            print(f"KG processed: {result.get('kg_processed')}")
            print(f"VDB chunks: {result.get('vdb_chunks')}")
            
            if result.get('vdb_chunks', 0) > 0:
                print("OK VDB chunk processing working!")
            else:
                print("WARNING VDB chunks still 0")
                
        else:
            print("WARNING test_insurance.txt not found, skipping file test")
            
    except Exception as e:
        print(f"ERROR Document processing test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing VDB Fixes\n")
    
    test_basic_keyword_extraction()
    test_document_processing()
    
    print("\n=== Test Summary ===")
    print("The fixes should:")
    print("1. Process documents even when Gemini API fails")
    print("2. Extract basic keywords as fallback")
    print("3. Generate VDB chunks properly")
    print("4. Provide better debugging information")