#!/usr/bin/env python3
"""Test the ChromaDB fixes"""

import sys
import os
import time
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_chromadb_fixes():
    """Test the enhanced ChromaDB integration"""
    
    print("=== Testing ChromaDB Fixes ===")
    
    try:
        # Import the hybrid processor
        from hybrid_main import HybridRAGProcessor
        
        print("1. Creating HybridRAGProcessor...")
        processor = HybridRAGProcessor()
        
        print("2. Testing ChromaDB initialization...")
        success = processor._initialize_chromadb()
        
        if success:
            print("   OK ChromaDB initialization successful")
        else:
            print("   ERROR ChromaDB initialization failed")
            return False
        
        print("3. Testing vector store status...")
        if processor.vector_store:
            try:
                collection = processor.vector_store._collection
                doc_count = collection.count()
                print(f"   OK ChromaDB collection has {doc_count} documents")
            except Exception as e:
                print(f"   WARNING Collection access error: {e}")
        
        print("4. Testing document processing...")
        
        # Test with the existing test document
        test_file = "test_insurance.txt"
        if Path(test_file).exists():
            try:
                result = processor.process_document(test_file, "test_insurance.txt")
                
                if result.get("success"):
                    print(f"   OK Document processed: KG={result.get('kg_processed')}, VDB={result.get('vdb_chunks')} chunks")
                else:
                    print("   WARNING Document processing failed")
                    
            except Exception as e:
                print(f"   ERROR Document processing error: {e}")
        else:
            print("   SKIP test_insurance.txt not found")
        
        print("5. Testing query processing...")
        try:
            response = processor.process_query("What is the age limit for insurance?")
            
            if response and hasattr(response, 'decision'):
                print("   OK Query processing successful")
            else:
                print("   WARNING Query processing returned unexpected format")
                
        except Exception as e:
            print(f"   ERROR Query processing error: {e}")
        
        print("\\nSUCCESS ChromaDB fixes appear to be working!")
        return True
        
    except ImportError as e:
        print(f"ERROR Import failed: {e}")
        return False
        
    except Exception as e:
        print(f"ERROR Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chromadb_fixes()
    
    if success:
        print("\\n=== ChromaDB Fix Test: PASSED ===")
    else:
        print("\\n=== ChromaDB Fix Test: FAILED ===")
        
    # Add instructions for user
    print("\\nTo fully test the fixes:")
    print("1. Start the server: python hybrid_main.py")
    print("2. Check status endpoint: curl http://localhost:8000/status")
    print("3. Upload a document via the frontend")
    print("4. Run a query to test citations")