#!/usr/bin/env python3
"""Debug ChromaDB initialization issues"""

import os
import sys
from pathlib import Path
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_chromadb_initialization():
    """Test ChromaDB initialization step by step"""
    
    print("=== ChromaDB Initialization Debug ===")
    
    try:
        # Step 1: Import ChromaDB
        print("1. Testing ChromaDB import...")
        import chromadb
        print(f"   OK ChromaDB version: {chromadb.__version__}")
        
        # Step 2: Import sentence transformers
        print("2. Testing sentence-transformers import...")
        from sentence_transformers import SentenceTransformer
        print("   OK Sentence transformers imported")
        
        # Step 3: Import Langchain components
        print("3. Testing Langchain imports...")
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("   OK Langchain components imported")
        
        # Step 4: Create persist directory
        print("4. Creating persist directory...")
        persist_dir = Path("data/chroma_db")
        persist_dir.mkdir(parents=True, exist_ok=True)
        print(f"   OK Directory created: {persist_dir.absolute()}")
        
        # Step 5: Initialize embeddings
        print("5. Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("   OK Embeddings initialized")
        
        # Step 6: Test embedding function
        print("6. Testing embedding function...")
        test_embedding = embeddings.embed_query("test document")
        print(f"   OK Embedding dimension: {len(test_embedding)}")
        
        # Step 7: Initialize ChromaDB with Langchain
        print("7. Initializing ChromaDB with Langchain...")
        vector_store = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name="test_collection"
        )
        print("   OK ChromaDB initialized with Langchain")
        
        # Step 8: Test basic operations
        print("8. Testing basic operations...")
        
        # Add a test document
        test_docs = ["This is a test insurance document about health coverage."]
        test_metadatas = [{"document_id": "test_doc", "chunk_id": "chunk_1"}]
        
        vector_store.add_texts(
            texts=test_docs,
            metadatas=test_metadatas,
            ids=["test_id_1"]
        )
        print("   OK Test document added")
        
        # Query the document
        results = vector_store.similarity_search("health insurance", k=1)
        print(f"   OK Query successful, found {len(results)} results")
        
        if results:
            print(f"   Result: Result: {results[0].page_content[:50]}...")
        
        # Step 9: Check collection stats
        print("9. Checking collection statistics...")
        collection = vector_store._collection
        count = collection.count()
        print(f"   OK Collection has {count} documents")
        
        print("\nSUCCESS ChromaDB initialization completely successful!")
        return True
        
    except Exception as e:
        print(f"\nFAILED ChromaDB initialization failed at step: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def test_direct_chromadb():
    """Test direct ChromaDB without Langchain"""
    
    print("\n=== Direct ChromaDB Test ===")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Initialize ChromaDB client directly
        client = chromadb.PersistentClient(
            path="data/chroma_db_direct",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        collection = client.get_or_create_collection(
            name="test_direct",
            metadata={"description": "Test collection"}
        )
        
        # Add test document
        collection.add(
            documents=["Test insurance policy document"],
            metadatas=[{"document_id": "test"}],
            ids=["test_1"]
        )
        
        # Query
        results = collection.query(
            query_texts=["insurance policy"],
            n_results=1
        )
        
        print(f"OK Direct ChromaDB works! Found {len(results['documents'][0])} results")
        return True
        
    except Exception as e:
        print(f"FAILED Direct ChromaDB failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    
    # Test both approaches
    langchain_success = test_chromadb_initialization()
    direct_success = test_direct_chromadb()
    
    print(f"\n=== SUMMARY ===")
    print(f"Langchain ChromaDB: {'OK SUCCESS' if langchain_success else 'FAILED FAILED'}")
    print(f"Direct ChromaDB: {'OK SUCCESS' if direct_success else 'FAILED FAILED'}")
    
    if not langchain_success and not direct_success:
        print("\nFIX NEEDED ChromaDB needs fixing!")
    elif langchain_success:
        print("\nSUCCESS ChromaDB is working perfectly!")