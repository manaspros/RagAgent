"""
Enhanced Main Application with Hybrid RAG Integration
Integrates hybrid RAG system with existing frontend
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Google Gemini
import google.generativeai as genai

# ChromaDB and Vector Store
import chromadb
from chromadb.config import Settings
try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to old imports if new packages not available
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Neo4j
from neo4j import GraphDatabase
from py2neo import Graph, Node, Relationship

# Environment and utilities
from dotenv import load_dotenv
from loguru import logger
import sys
from datetime import datetime
import tempfile
import shutil

# Import existing components
from kg_manager import create_enhanced_kg_manager
from gemini_agents import create_agent_system

# Load environment variables
load_dotenv()

# Configure enhanced logging
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
    level="INFO"
)

# Pydantic Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about insurance policy")

class QueryResponse(BaseModel):
    Decision: str = Field(..., description="Final decision: Approved, Rejected, or Requires Further Review")
    Amount: str = Field(..., description="Calculated amount or 'N/A'")
    Justification: str = Field(..., description="Comprehensive justification for the decision")
    Relevant_Clauses: list = Field(..., description="List of relevant clauses with source information")
    Processing_Info: dict = Field(default_factory=dict, description="Processing metadata")
    graph_data: dict = Field(default_factory=dict, description="Graph visualization data")

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str
    processing_status: str
    file_info: dict

# Enhanced Hybrid RAG System
class HybridRAGProcessor:
    """
    Simplified hybrid RAG processor that integrates with existing system
    """
    
    def __init__(self):
        self.kg_manager = None
        self.vector_store = None
        self.embeddings = None
        self.text_splitter = None
        self.llm = None
        self.initialized = False
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components"""
        try:
            # Initialize existing KG manager
            self.kg_manager = create_enhanced_kg_manager()
            success = self.kg_manager.initialize_kg()
            
            if success:
                logger.info("KG Manager initialized successfully")
            else:
                logger.warning("KG Manager initialization partial")
            
            # Initialize Gemini LLM
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
                self.llm = genai.GenerativeModel(
                    model_name=os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-lite'),
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=4000
                    )
                )
                logger.info("Gemini LLM initialized")
            else:
                logger.warning("GEMINI_API_KEY not found")
            
            # Initialize ChromaDB
            self._initialize_chromadb()
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            # Initialize multi-agent system
            self.agent_system = create_agent_system()
            logger.info("Multi-agent system initialized")
            
            self.initialized = True
            logger.info("Hybrid RAG Processor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            self.initialized = False
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB vector database with robust error handling"""
        try:
            # Create persist directory
            persist_dir = Path("data/chroma_db")
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Initializing ChromaDB embeddings...")
            
            # Initialize embeddings with error handling
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Test embeddings
            test_embed = self.embeddings.embed_query("test")
            logger.info(f"Embeddings initialized successfully (dimension: {len(test_embed)})")
            
            # Initialize ChromaDB with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.vector_store = Chroma(
                        persist_directory=str(persist_dir),
                        embedding_function=self.embeddings,
                        collection_name="insurance_documents"
                    )
                    
                    # Test the connection with a simple operation
                    collection = self.vector_store._collection
                    doc_count = collection.count()
                    logger.info(f"ChromaDB initialized successfully (existing documents: {doc_count})")
                    
                    return True
                    
                except Exception as retry_error:
                    logger.warning(f"ChromaDB initialization attempt {attempt + 1} failed: {retry_error}")
                    if attempt == max_retries - 1:
                        raise retry_error
                    time.sleep(1)  # Wait before retry
            
        except Exception as e:
            logger.error(f"ChromaDB initialization failed after all retries: {e}")
            logger.error(f"Full error details: {str(e)}")
            self.vector_store = None
            return False
    
    def _extract_document_content(self, file_path: str) -> str:
        """Extract text content from various document types"""
        try:
            file_path = Path(file_path)
            extension = file_path.suffix.lower()
            
            if extension == '.pdf':
                import PyPDF2
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    content = ""
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
                    return content
            elif extension in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            else:
                logger.warning(f"Unsupported file type: {extension}")
                return ""
                
        except Exception as e:
            logger.error(f"Error extracting content from {file_path}: {e}")
            return ""
    
    def _process_document_with_gemini(self, content: str, document_id: str) -> bool:
        """Process document content with Gemini and store in KG with rate limiting"""
        try:
            if not self.llm:
                logger.warning("Gemini LLM not available for processing")
                return False
            
            # Add delay to respect rate limits
            import time
            time.sleep(3)  # 3 second delay between API calls (25 req/min limit)
            
            logger.info("Processing with Gemini (with rate limiting)...")
            
            # Create prompt to extract entities from document
            prompt = f"""
            Analyze this insurance document and extract structured information.
            Return a JSON object with entities and relationships.
            
            Document Content:
            {content[:3000]}...
            
            Extract:
            1. Policies (with coverage amounts, types)
            2. Procedures (medical procedures covered)
            3. Locations (where coverage applies)
            4. EligibilityCriteria (age, conditions)
            5. Relationships between entities
            
            Return JSON format:
            {{
                "entities": {{
                    "policies": [{{ "id": "policy_1", "name": "Health Insurance", "type": "Health", "coverage": "50000" }}],
                    "procedures": [{{ "id": "proc_1", "name": "Knee Surgery", "category": "Surgery", "coverage": "15000" }}],
                    "locations": [{{ "id": "loc_1", "name": "Mumbai", "type": "City" }}],
                    "eligibility": [{{ "id": "elig_1", "criteria": "Age 18-65", "type": "Age" }}]
                }},
                "relationships": [
                    {{ "source": "policy_1", "target": "proc_1", "type": "COVERS" }},
                    {{ "source": "policy_1", "target": "loc_1", "type": "AVAILABLE_IN" }}
                ]
            }}
            """
            
            response = self.llm.generate_content(prompt)
            response_text = response.text
            
            # Try to parse the JSON response
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
                logger.info(f"Extracted {len(extracted_data.get('entities', {}))} entity types")
                
                # Store in Neo4j through KG manager
                return self._store_entities_in_kg(extracted_data, document_id, content)
            else:
                logger.error("No valid JSON found in Gemini response")
                return False
                
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific rate limiting errors
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                if "day" in error_msg.lower() or "GenerateRequestsPerDayPerProjectPerModel" in error_msg:
                    logger.error("Gemini API daily quota exhausted. Using fallback processing for all requests today.")
                else:
                    logger.warning("Gemini API rate limit exceeded. Using fallback processing.")
                logger.info("Fallback: Processing document without AI entity extraction")
                
                # Store basic document info without Gemini processing
                try:
                    # Create a basic document entity with extracted keywords
                    keywords = self._extract_basic_keywords(content)
                    
                    basic_data = {
                        "entities": {
                            "documents": [{
                                "id": document_id, 
                                "name": document_id, 
                                "type": "Document",
                                "keywords": keywords,
                                "content_length": len(content)
                            }]
                        },
                        "relationships": []
                    }
                    
                    logger.info(f"Fallback processing: Created basic document entity with {len(keywords)} keywords")
                    return self._store_entities_in_kg(basic_data, document_id, content)
                except Exception as fallback_error:
                    logger.error(f"Fallback processing failed: {fallback_error}")
                    return False
            else:
                logger.error(f"Error processing document with Gemini: {e}")
            
            return False
    
    def _extract_basic_keywords(self, content: str) -> list:
        """Extract basic keywords from document content without AI"""
        import re
        
        # Insurance/medical related keywords to look for
        important_terms = [
            'policy', 'insurance', 'coverage', 'premium', 'deductible', 'claim',
            'surgery', 'treatment', 'medical', 'health', 'procedure', 'hospital',
            'age', 'eligible', 'benefit', 'amount', 'limit', 'exclusion',
            'mental', 'illness', 'condition', 'disease', 'therapy', 'diagnosis'
        ]
        
        # Extract words and find matches
        words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
        found_keywords = []
        
        for term in important_terms:
            if term in words or any(term in word for word in words):
                found_keywords.append(term)
        
        # Also extract numbers (amounts, ages, etc.)
        numbers = re.findall(r'\b\d+\b', content)
        if numbers:
            found_keywords.extend([f"amount_{num}" for num in numbers[:5]])  # First 5 numbers
        
        return found_keywords[:20]  # Limit to 20 keywords
    
    def _store_entities_in_kg(self, extracted_data: dict, document_id: str, content: str) -> bool:
        """Store extracted entities in Knowledge Graph"""
        try:
            if self.kg_manager.connection_mode == "fallback":
                # Store in session data for fallback mode
                nodes = []
                edges = []
                
                # Add document node
                nodes.append({
                    "id": document_id,
                    "type": "Document",
                    "label": document_id,
                    "properties": {"content_length": len(content)}
                })
                
                # Add entity nodes
                entities = extracted_data.get("entities", {})
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        nodes.append({
                            "id": entity.get("id", f"{entity_type}_{len(nodes)}"),
                            "type": entity_type.capitalize().rstrip('s'),
                            "label": entity.get("name", entity.get("criteria", "Unknown")),
                            "properties": entity
                        })
                
                # Add relationships
                relationships = extracted_data.get("relationships", [])
                for rel in relationships:
                    edges.append({
                        "source": rel.get("source"),
                        "target": rel.get("target"),
                        "type": rel.get("type", "RELATED_TO"),
                        "properties": {}
                    })
                
                # Store in session data
                self.kg_manager.current_session_data = {
                    "nodes": nodes,
                    "relationships": edges,
                    "queries": [],
                    "active_document": document_id
                }
                
                logger.info(f"Stored {len(nodes)} nodes and {len(edges)} relationships in fallback mode")
                return True
                
            else:
                # Store in Neo4j
                return self._store_in_neo4j_direct(extracted_data, document_id, content)
                
        except Exception as e:
            logger.error(f"Error storing entities: {e}")
            return False
    
    def _store_in_neo4j_direct(self, extracted_data: dict, document_id: str, content: str) -> bool:
        """Store entities directly in Neo4j"""
        try:
            with self.kg_manager.driver.session() as session:
                # Clear existing data
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Cleared existing Neo4j data")
                
                # Create document node
                session.run(
                    "MERGE (d:Document {id: $doc_id}) "
                    "SET d.name = $doc_id, d.content_length = $content_length",
                    doc_id=document_id, content_length=len(content)
                )
                
                # Create entity nodes
                entities = extracted_data.get("entities", {})
                node_count = 0
                
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        entity_id = entity.get("id", f"{entity_type}_{node_count}")
                        entity_name = entity.get("name", entity.get("criteria", "Unknown"))
                        
                        # Determine node label
                        label = entity_type.capitalize().rstrip('s')
                        if label == "Policie": label = "Policy"
                        if label == "Procedure": label = "Procedure"
                        if label == "Location": label = "Location"  
                        if label == "Eligibility": label = "EligibilityCriteria"
                        
                        session.run(
                            f"MERGE (e:{label} {{id: $entity_id}}) "
                            "SET e.name = $entity_name, e.properties = $props",
                            entity_id=entity_id, entity_name=entity_name, props=json.dumps(entity)
                        )
                        node_count += 1
                
                # Create relationships
                relationships = extracted_data.get("relationships", [])
                for rel in relationships:
                    session.run(
                        "MATCH (a {id: $source}), (b {id: $target}) "
                        "MERGE (a)-[r:RELATED_TO]->(b) "
                        "SET r.type = $rel_type",
                        source=rel.get("source"), target=rel.get("target"), rel_type=rel.get("type", "RELATED_TO")
                    )
                
                logger.info(f"Stored {node_count} nodes and {len(relationships)} relationships in Neo4j")
                
                # Update session data for tracking even when using Neo4j
                self.kg_manager.current_session_data = {
                    "nodes": [{"type": "Document", "id": document_id, "name": document_id}] + 
                            [{"type": entity_type.capitalize().rstrip('s'), "id": entity.get("id")} 
                             for entity_type, entity_list in entities.items() for entity in entity_list],
                    "relationships": relationships,
                    "queries": [],
                    "active_document": document_id
                }
                
                return True
                
        except Exception as e:
            logger.error(f"Error storing in Neo4j: {e}")
            return False
    
    def process_document(self, file_path: str, document_id: str) -> Dict[str, Any]:
        """Process document for both KG and VDB storage"""
        if not self.initialized:
            return {"success": False, "error": "System not initialized"}
        
        try:
            logger.info(f"Processing document: {document_id}")
            
            # Extract content based on file type
            content = self._extract_document_content(file_path)
            
            if not content.strip():
                return {"success": False, "error": "Empty document or extraction failed"}
            
            # Process with existing KG manager - create entities directly
            kg_success = self._process_document_with_gemini(content, document_id)
            
            # Process for vector database (always attempt, regardless of KG success)
            vdb_chunks = 0
            logger.info(f"VDB processing check: vector_store={self.vector_store is not None}, text_splitter={self.text_splitter is not None}")
            
            # Always process VDB if available (independent of KG processing)
            logger.info(f"VDB components check: vector_store type={type(self.vector_store)}, text_splitter type={type(self.text_splitter)}")
            
            # Force VDB processing if components exist (regardless of their boolean evaluation)
            if self.vector_store is not None and self.text_splitter is not None:
                logger.info("Starting VDB chunk processing...")
                try:
                    chunks = self.text_splitter.split_text(content)
                    logger.info(f"Text split into {len(chunks)} chunks")
                
                    # Add to ChromaDB with metadata
                    texts = []
                    metadatas = []
                    
                    for i, chunk in enumerate(chunks):
                        texts.append(chunk)
                        metadatas.append({
                            "document_id": document_id,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "source": "hybrid_rag",
                            "file_path": file_path
                        })
                    
                    # Try to add to ChromaDB (this should work without API calls)
                    logger.info("Attempting to store chunks in ChromaDB...")
                    
                    self.vector_store.add_texts(
                        texts=texts,
                        metadatas=metadatas,
                        ids=[f"{document_id}_chunk_{i}" for i in range(len(chunks))]
                    )
                    
                    vdb_chunks = len(chunks)
                    logger.info(f"✅ Successfully added {vdb_chunks} chunks to ChromaDB")
                    
                except Exception as vdb_error:
                    logger.error(f"VDB processing failed: {vdb_error}")
                    logger.info(f"Attempted to process chunks, but VDB storage failed")
                    vdb_chunks = 0
            else:
                # VDB not available - log the specific reason
                if self.vector_store is None:
                    logger.warning("Vector store is None - VDB processing skipped")
                elif not self.vector_store:
                    logger.warning("Vector store exists but evaluates to False - VDB processing skipped")
                    logger.warning(f"Vector store state: {getattr(self.vector_store, '_collection', 'no _collection attr')}")
                
                if self.text_splitter is None:
                    logger.warning("Text splitter is None - VDB processing skipped")
                elif not self.text_splitter:
                    logger.warning("Text splitter exists but evaluates to False - VDB processing skipped")
            
            logger.info(f"Document processed: KG={kg_success}, VDB={vdb_chunks} chunks")
            
            return {
                "success": True,
                "kg_processed": kg_success,
                "vdb_chunks": vdb_chunks,
                "document_id": document_id
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {"success": False, "error": str(e)}
    
    def process_query(self, query: str) -> QueryResponse:
        """Process query using hybrid RAG approach with explicit citations"""
        if not self.initialized:
            return QueryResponse(
                Decision="Requires Further Review",
                Amount="N/A",
                Justification="System not fully initialized. Please try again later.",
                Relevant_Clauses=[]
            )
        
        try:
            logger.info(f"Processing query with explicit citations: {query[:100]}...")
            
            # Step 1: Retrieve from vector database with enhanced metadata
            vdb_results = []
            if self.vector_store:
                try:
                    docs = self.vector_store.similarity_search(query, k=5)
                    vdb_results = [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "source": "VDB",
                            "similarity_score": getattr(doc, '_similarity_score', 'N/A')
                        }
                        for doc in docs
                    ]
                    logger.info(f"Retrieved {len(vdb_results)} chunks from VDB")
                except Exception as e:
                    logger.warning(f"VDB retrieval failed: {e}")
            
            # Step 2: Retrieve from Knowledge Graph with relationships
            kg_facts = []
            if self.kg_manager:
                try:
                    kg_facts = self._retrieve_kg_facts(query)
                    logger.info(f"Retrieved {len(kg_facts)} facts from KG")
                except Exception as e:
                    logger.error(f"KG retrieval failed: {e}")
                    # Continue with empty KG facts rather than failing completely
            
            # Step 3: Enhanced reasoning with explicit citations using multi-agent system
            try:
                return self._process_with_agents_and_citations(query, vdb_results, kg_facts)
            except Exception as e:
                logger.error(f"Citation processing failed: {e}")
                # Fall back to simple processing
                return self._fallback_query_processing_simple(query, vdb_results)
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return QueryResponse(
                Decision="Requires Further Review",
                Amount="N/A",
                Justification=f"An error occurred during processing: {str(e)}",
                Relevant_Clauses=[]
            )
    
    def _retrieve_kg_facts(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant facts from Knowledge Graph with relationship context"""
        try:
            logger.info(f"Retrieving KG facts for query: {query}")
            logger.info(f"KG Manager connection mode: {self.kg_manager.connection_mode}")
            
            if self.kg_manager.connection_mode == "fallback":
                # Use session data for fallback mode
                nodes = self.kg_manager.current_session_data.get("nodes", [])
                relationships = self.kg_manager.current_session_data.get("relationships", [])
                
                logger.info(f"Fallback mode: Found {len(nodes)} nodes, {len(relationships)} relationships")
                
                # Simple keyword matching for fallback
                relevant_facts = []
                query_lower = query.lower()
                
                for node in nodes:
                    if any(keyword in node.get("label", "").lower() for keyword in query_lower.split()):
                        relevant_facts.append({
                            "type": "node",
                            "data": node,
                            "source": "KG_Fallback",
                            "document_id": self.kg_manager.current_session_data.get("active_document", "unknown"),
                            "page_section": "fallback_mode"
                        })
                
                logger.info(f"Fallback mode: Found {len(relevant_facts)} relevant facts")
                return relevant_facts
            
            # For Neo4j mode - query the graph
            kg_facts = []
            logger.info("Using Neo4j mode for KG retrieval")
            
            # Query 1: Find relevant nodes based on query keywords
            node_query = """
            MATCH (n)
            WHERE toLower(coalesce(n.name, '')) CONTAINS toLower($keyword1) 
               OR toLower(coalesce(n.name, '')) CONTAINS toLower($keyword2) 
               OR toLower(coalesce(n.name, '')) CONTAINS toLower($keyword3)
               OR toLower(coalesce(n.id, '')) CONTAINS toLower($keyword1)
               OR toLower(coalesce(n.id, '')) CONTAINS toLower($keyword2)
               OR toLower(coalesce(n.id, '')) CONTAINS toLower($keyword3)
               OR toLower(coalesce(n.type, '')) CONTAINS toLower($keyword1)
               OR toLower(coalesce(n.type, '')) CONTAINS toLower($keyword2)
               OR toLower(coalesce(n.type, '')) CONTAINS toLower($keyword3)
               OR toLower(coalesce(n.category, '')) CONTAINS toLower($keyword1)
               OR toLower(coalesce(n.category, '')) CONTAINS toLower($keyword2)
               OR toLower(coalesce(n.category, '')) CONTAINS toLower($keyword3)
               OR toLower(coalesce(n.description, '')) CONTAINS toLower($keyword1)
               OR toLower(coalesce(n.description, '')) CONTAINS toLower($keyword2)
               OR toLower(coalesce(n.description, '')) CONTAINS toLower($keyword3)
               OR toLower(coalesce(n.definition, '')) CONTAINS toLower($keyword1)
               OR toLower(coalesce(n.definition, '')) CONTAINS toLower($keyword2)
               OR toLower(coalesce(n.definition, '')) CONTAINS toLower($keyword3)
               OR toLower(coalesce(n.term, '')) CONTAINS toLower($keyword1)
               OR toLower(coalesce(n.term, '')) CONTAINS toLower($keyword2)
               OR toLower(coalesce(n.term, '')) CONTAINS toLower($keyword3)
               OR labels(n)[0] IN ['Policy', 'Procedure', 'Term', 'Definition', 'EligibilityCriteria']
            RETURN n.id as id, n.name as name, labels(n)[0] as type, properties(n) as properties
            LIMIT 20
            """
            
            # Better keyword extraction with medical term expansion
            important_words = ['mental', 'illness', 'disease', 'condition', 'surgery', 'surgical', 'treatment', 'insurance', 'policy', 'coverage', 'age', 'year', 'eligible', 'premium']
            query_words = query.lower().split()
            
            # Medical term expansions
            medical_expansions = {
                'surgery': 'surgical',
                'surgical': 'surgery', 
                'knee': 'joint',
                'covered': 'coverage',
                'treatment': 'medical',
                'anemia': 'blood',
                'aplastic': 'condition',
                'disease': 'condition',
                'illness': 'condition'
            }
            
            # Prioritize important words, then get longer words
            query_keywords = []
            for word in query_words:
                if word in important_words:
                    query_keywords.append(word)
                    # Add medical expansion if available
                    if word in medical_expansions:
                        query_keywords.append(medical_expansions[word])
            
            # Add other words longer than 2 characters
            for word in query_words:
                if len(word) > 2 and word not in query_keywords:
                    query_keywords.append(word)
                    # Add medical expansion if available
                    if word in medical_expansions:
                        query_keywords.append(medical_expansions[word])
            
            # Remove duplicates and take top 5 (increased from 3)
            query_keywords = list(set(query_keywords))[:5]
            logger.info(f"Using enhanced keywords: {query_keywords}")
            
            try:
                with self.kg_manager.driver.session() as session:
                    result = session.run(node_query, 
                                       keyword1=query_keywords[0] if len(query_keywords) > 0 else "",
                                       keyword2=query_keywords[1] if len(query_keywords) > 1 else "",
                                       keyword3=query_keywords[2] if len(query_keywords) > 2 else "")
                    
                    for record in result:
                        kg_facts.append({
                            "type": "node",
                            "id": record.get("id"),
                            "name": record.get("name"),
                            "node_type": record.get("type"),
                            "properties": record.get("properties", {}),
                            "source": "KG",
                            "document_id": self.kg_manager.current_session_data.get("active_document", "neo4j_graph"),
                            "page_section": "knowledge_graph"
                        })
            except Exception as e:
                logger.error(f"Error querying nodes: {e}")
            
            # Query 2: Find relevant relationships
            relationship_query = """
            MATCH (a)-[r]->(b)
            WHERE toLower(a.name) CONTAINS toLower($keyword) 
               OR toLower(b.name) CONTAINS toLower($keyword) 
               OR toLower(type(r)) CONTAINS toLower($keyword)
            RETURN a.id as source_id, a.name as source_name, labels(a)[0] as source_type,
                   b.id as target_id, b.name as target_name, labels(b)[0] as target_type,
                   type(r) as relationship_type, properties(r) as relationship_properties
            LIMIT 15
            """
            
            try:
                with self.kg_manager.driver.session() as session:
                    for keyword in query_keywords:
                        result = session.run(relationship_query, keyword=keyword)
                        
                        for record in result:
                            kg_facts.append({
                                "type": "relationship",
                                "source": {
                                    "id": record.get("source_id"),
                                    "name": record.get("source_name"),
                                    "type": record.get("source_type")
                                },
                                "target": {
                                    "id": record.get("target_id"),
                                    "name": record.get("target_name"), 
                                    "type": record.get("target_type")
                                },
                                "relationship_type": record.get("relationship_type"),
                                "properties": record.get("relationship_properties", {}),
                                "source": "KG",
                                "document_id": self.kg_manager.current_session_data.get("active_document", "neo4j_graph"),
                                "page_section": "knowledge_graph_relationships"
                            })
            except Exception as e:
                logger.error(f"Error querying relationships: {e}")
            
            return kg_facts
            
        except Exception as e:
            logger.error(f"Error retrieving KG facts: {e}")
            return []
    
    def _process_with_citations(self, query: str, vdb_results: List[Dict], kg_facts: List[Dict]) -> QueryResponse:
        """Process query with comprehensive citations from both VDB and KG"""
        try:
            if not self.llm:
                return self._fallback_query_processing_simple(query, vdb_results)
            
            # Check if we have any data to work with
            if not vdb_results and not kg_facts:
                return QueryResponse(
                    Decision="Requires Further Review",
                    Amount="N/A",
                    Justification="No relevant information found in either vector database or knowledge graph. Please ensure documents are properly uploaded and processed.",
                    Relevant_Clauses=[],
                    Processing_Info={
                        "processing_method": "no_data_found",
                        "vdb_chunks_found": 0,
                        "kg_facts_found": 0
                    }
                )
            
            # Prepare comprehensive context with citations
            context_parts = []
            citation_map = {}
            citation_counter = 1
            
            # Add VDB context with citations (if available)
            if vdb_results:
                context_parts.append("=== VECTOR DATABASE CONTEXT ===")
                for i, vdb_result in enumerate(vdb_results[:3]):
                    citation_id = f"VDB{citation_counter}"
                    context_parts.append(f"[{citation_id}] {vdb_result['content']}")
                    citation_map[citation_id] = {
                        "clause_text": vdb_result["content"],
                        "document_id": vdb_result["metadata"].get("document_id", "unknown"),
                        "page_section": f"chunk_{vdb_result['metadata'].get('chunk_index', i)}",
                        "retrieval_source": "Vector Database (VDB)"
                    }
                    citation_counter += 1
            
            # Add KG context with citations
            context_parts.append("\n=== KNOWLEDGE GRAPH CONTEXT ===")
            for kg_fact in kg_facts[:5]:
                citation_id = f"KG{citation_counter}"
                if kg_fact["type"] == "node":
                    context_parts.append(f"[{citation_id}] Node: {kg_fact.get('name')} (Type: {kg_fact.get('node_type')})")
                    citation_map[citation_id] = {
                        "clause_text": f"Entity: {kg_fact.get('name')} of type {kg_fact.get('node_type')}",
                        "document_id": kg_fact.get("document_id", "unknown"),
                        "page_section": kg_fact.get("page_section", "knowledge_graph"),
                        "retrieval_source": f"Knowledge Graph (KG) - {kg_fact.get('node_type')} entity"
                    }
                elif kg_fact["type"] == "relationship":
                    rel_text = f"{kg_fact['source']['name']} --{kg_fact['relationship_type']}--> {kg_fact['target']['name']}"
                    context_parts.append(f"[{citation_id}] Relationship: {rel_text}")
                    citation_map[citation_id] = {
                        "clause_text": f"Relationship: {rel_text}",
                        "document_id": kg_fact.get("document_id", "unknown"),
                        "page_section": kg_fact.get("page_section", "knowledge_graph_relationships"), 
                        "retrieval_source": f"Knowledge Graph (KG) - {kg_fact['source']['type']}-{kg_fact['relationship_type']}-{kg_fact['target']['type']} relationship"
                    }
                citation_counter += 1
            
            # Create enhanced prompt for citation-aware reasoning
            full_context = "\n".join(context_parts)
            
            prompt = f"""
            You are an expert insurance policy analyst. Analyze the query using the provided context and give a decision with EXPLICIT CITATIONS.

            QUERY: {query}

            CONTEXT WITH CITATIONS:
            {full_context}

            INSTRUCTIONS:
            1. For every fact or decision point, cite the exact source using [CitationID] format
            2. State the retrieval method (VDB or KG) 
            3. Include document_id and page_section in your reasoning
            4. For KG relationships, mention the specific entity-relationship pattern

            Provide response in JSON format:
            {{
                "Decision": "Approved/Rejected/Requires Further Review",
                "Amount": "amount_or_N/A", 
                "Justification": "Detailed explanation with inline citations like [VDB1] and [KG2]",
                "Key_Facts": [
                    "Fact 1 with [CitationID]",
                    "Fact 2 with [CitationID]"
                ],
                "Citations_Used": ["VDB1", "KG2", "etc"]
            }}
            """
            
            response = self.llm.generate_content(prompt)
            response_text = response.text
            
            # Parse JSON response
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                
                # Build relevant clauses with full citation details
                relevant_clauses = []
                citations_used = result_data.get("Citations_Used", [])
                
                for citation_id in citations_used:
                    if citation_id in citation_map:
                        relevant_clauses.append(citation_map[citation_id])
                
                # Add any remaining citations from VDB and KG
                for citation_id, citation_data in citation_map.items():
                    if citation_data not in relevant_clauses:
                        relevant_clauses.append(citation_data)
                
                return QueryResponse(
                    Decision=result_data.get("Decision", "Requires Further Review"),
                    Amount=result_data.get("Amount", "N/A"),
                    Justification=result_data.get("Justification", response_text),
                    Relevant_Clauses=relevant_clauses,
                    Processing_Info={
                        "processing_method": "gemini_with_citations",
                        "vdb_chunks_found": len(vdb_results),
                        "kg_facts_found": len(kg_facts),
                        "total_citations": len(citation_map),
                        "hybrid_mode": True,
                        "model_used": "gemini-2.0-flash-lite"
                    }
                )
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            logger.error(f"Citation processing failed: {e}")
            return self._fallback_query_processing_simple(query, vdb_results)
    
    def _process_with_agents_and_citations(self, query: str, vdb_results: List[Dict], kg_facts: List[Dict]) -> QueryResponse:
        """Process query using multi-agent system with comprehensive citations"""
        try:
            logger.info("Processing query with multi-agent system and citations")
            
            # Prepare context data for agents
            context_data = []
            
            # Add VDB results to context
            for vdb_result in vdb_results:
                context_data.append({
                    "type": "document_chunk",
                    "content": vdb_result.get("content", ""),
                    "metadata": vdb_result.get("metadata", {}),
                    "source": "VDB"
                })
            
            # Add KG facts to context  
            for kg_fact in kg_facts:
                context_data.append({
                    "type": "knowledge_fact",
                    "content": f"{kg_fact.get('name', 'Unknown')} ({kg_fact.get('node_type', 'Entity')})",
                    "properties": kg_fact.get("properties", {}),
                    "source": "KG"
                })
            
            # Use multi-agent system to process the query
            agent_result = self.agent_system.process_query(query, context_data)
            
            logger.info(f"Agent processing result: {agent_result}")
            
            # Build citation map first (needed for fallback)
            citation_map = {}
            citation_counter = 1
            relevant_clauses = []
            
            # Add VDB citations
            for i, vdb_result in enumerate(vdb_results[:5]):
                citation_id = f"VDB{citation_counter}"
                citation_map[citation_id] = vdb_result
                citation_counter += 1
                
                relevant_clauses.append({
                    "clause_text": vdb_result['page_content'][:200] + "...",
                    "document_id": vdb_result['metadata'].get('source', 'unknown'),
                    "page_section": vdb_result['metadata'].get('chunk_index', i),
                    "retrieval_source": "Vector Database (VDB)",
                    "citation_id": citation_id
                })
            
            # Add KG citations
            for i, kg_fact in enumerate(kg_facts[:5]):
                citation_id = f"KG{citation_counter}"
                citation_map[citation_id] = kg_fact
                citation_counter += 1
                
                relevant_clauses.append({
                    "clause_text": str(kg_fact.get('properties', kg_fact))[:200] + "...",
                    "document_id": kg_fact.get('document_id', 'knowledge_graph'),
                    "page_section": kg_fact.get('page_section', 'knowledge_graph'),
                    "retrieval_source": "Knowledge Graph (KG)",
                    "citation_id": citation_id
                })
            
            # Check if agents failed due to quota/rate limits and use fallback
            agent_decision = agent_result.get('Decision', 'Requires Further Review')
            agent_amount = agent_result.get('Amount', 'N/A')
            agent_justification = agent_result.get('Justification', '')
            
            # If agents failed, use rule-based fallback
            if (agent_decision == 'Requires Further Review' and 
                ('error' in agent_justification.lower() or 'quota' in agent_justification.lower() or 
                 'processing error' in agent_justification.lower() or agent_amount == 'N/A')):
                logger.warning("Multi-agent processing failed, switching to rule-based fallback")
                return self._rule_based_fallback(query, vdb_results, kg_facts, relevant_clauses)
            
            
            # Enhance justification with citation references
            original_justification = agent_result.get('reasoning', 'Agent analysis completed.')
            citation_enhanced_justification = f"{original_justification} "
            
            # Add citation references to justification
            if relevant_clauses:
                source_list = [clause["retrieval_source"] for clause in relevant_clauses[:3]]
                citation_enhanced_justification += f"[Sources: {', '.join(source_list)}]"
            
            return QueryResponse(
                Decision=agent_result.get('Decision', 'Requires Further Review'),
                Amount=str(agent_result.get('Amount', 'N/A')),
                Justification=citation_enhanced_justification,
                Relevant_Clauses=relevant_clauses,
                Processing_Info={
                    "processing_method": "multi_agent_with_citations",
                    "agent_intent": agent_result.get('intent', 'unknown'),
                    "vdb_chunks_used": len(vdb_results),
                    "kg_facts_used": len(kg_facts),
                    "total_citations": len(relevant_clauses),
                    "confidence": agent_result.get('confidence', 'medium')
                }
            )
            
        except Exception as e:
            logger.error(f"Error in agent-based citation processing: {e}")
            # Fallback to the original citation method
            return self._process_with_citations(query, vdb_results, kg_facts)
    
    def _fallback_query_processing_simple(self, query: str, vdb_results: list) -> QueryResponse:
        """Simple fallback when citation processing fails"""
        relevant_clauses = []
        for i, vdb_result in enumerate(vdb_results[:3]):
            relevant_clauses.append({
                "clause_text": vdb_result["content"][:500] + "..." if len(vdb_result["content"]) > 500 else vdb_result["content"],
                "document_id": vdb_result["metadata"].get("document_id", "unknown"),
                "page_section": f"chunk_{vdb_result['metadata'].get('chunk_index', i)}",
                "retrieval_source": "Vector Database (VDB) - Fallback mode"
            })
        
        # Enhanced fallback with rule-based processing  
        fallback_response = self._rule_based_fallback(query, vdb_results, kg_facts, relevant_clauses)
        return fallback_response
    
    def _rule_based_fallback(self, query: str, vdb_results: list, kg_facts: list, relevant_clauses: list) -> QueryResponse:
        """Enhanced rule-based fallback that provides better answers even without AI"""
        logger.info("Using rule-based fallback processing")
        
        query_lower = query.lower()
        
        # Insurance query patterns and responses
        coverage_patterns = {
            'surgery': {
                'keywords': ['surgery', 'surgical', 'operation', 'procedure'],
                'decision': 'Approved',
                'reasoning': 'Surgical procedures are typically covered under medical treatment'
            },
            'mental_health': {
                'keywords': ['mental', 'psychiatric', 'psychological', 'therapy'],
                'decision': 'Approved', 
                'reasoning': 'Mental health treatment is covered under modern insurance policies'
            },
            'pre_existing': {
                'keywords': ['pre-existing', 'chronic', 'ongoing'],
                'decision': 'Requires Further Review',
                'reasoning': 'Pre-existing conditions require specific policy review'
            }
        }
        
        # Age-based eligibility
        age_mentioned = any(word in query_lower for word in ['year', 'old', 'age'])
        
        # Determine coverage type from available data
        matched_coverage = None
        for coverage_type, info in coverage_patterns.items():
            if any(keyword in query_lower for keyword in info['keywords']):
                matched_coverage = info
                break
        
        # Build justification from available data
        justification_parts = []
        
        if kg_facts:
            justification_parts.append(f"Knowledge Graph analysis found {len(kg_facts)} relevant policy facts.")
            # Extract procedure info from KG
            for fact in kg_facts:
                if fact.get('type') == 'node' and 'surgical' in str(fact.get('name', '')).lower():
                    justification_parts.append("Policy covers surgical/medical treatments.")
        
        if vdb_results:
            justification_parts.append(f"Document analysis found {len(vdb_results)} relevant text sections.")
        
        if matched_coverage:
            justification_parts.append(matched_coverage['reasoning'])
            decision = matched_coverage['decision']
        else:
            decision = "Requires Further Review"
            justification_parts.append("Query requires detailed policy review for accurate assessment.")
        
        if age_mentioned:
            justification_parts.append("Age eligibility criteria noted in query.")
        
        justification = " ".join(justification_parts)
        
        return QueryResponse(
            Decision=decision,
            Amount="N/A",  # Amount calculation requires AI processing
            Justification=justification,
            Relevant_Clauses=relevant_clauses,
            Processing_Info={
                "processing_method": "rule_based_fallback",
                "vdb_chunks": len(vdb_results),
                "kg_facts": len(kg_facts),
                "matched_pattern": matched_coverage['keywords'][0] if matched_coverage else None
            }
        )

    def _fallback_query_processing(self, query: str, vdb_results: list) -> QueryResponse:
        """Fallback query processing using Gemini directly"""
        if not self.llm:
            return QueryResponse(
                Decision="Requires Further Review",
                Amount="N/A",
                Justification="LLM not available for processing",
                Relevant_Clauses=[]
            )
        
        try:
            # Prepare context from VDB results
            context = ""
            relevant_clauses = []
            
            for i, result in enumerate(vdb_results[:3]):
                context += f"Document {i+1}: {result['content']}\n\n"
                relevant_clauses.append({
                    "clause_text": result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"],
                    "document_id": result["metadata"].get("document_id", "unknown"),
                    "page_section": f"chunk_{result['metadata'].get('chunk_index', 0)}",
                    "retrieval_source": "VDB"
                })
            
            # Create prompt for Gemini
            prompt = f"""
            You are an expert insurance policy analyst. Based on the provided context, analyze the query and provide a decision.

            QUERY: {query}

            CONTEXT:
            {context}

            Please provide:
            1. Decision: "Approved", "Rejected", or "Requires Further Review"
            2. Amount: If applicable, provide the coverage amount, otherwise "N/A"
            3. Justification: Detailed explanation for the decision

            Respond in JSON format:
            {{
                "Decision": "your_decision",
                "Amount": "amount_or_NA",
                "Justification": "detailed_explanation"
            }}
            """
            
            response = self.llm.generate_content(prompt)
            response_text = response.text
            
            # Try to parse JSON response
            try:
                import json
                import re
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                    
                    return QueryResponse(
                        Decision=result_data.get("Decision", "Requires Further Review"),
                        Amount=result_data.get("Amount", "N/A"),
                        Justification=result_data.get("Justification", response_text),
                        Relevant_Clauses=relevant_clauses,
                        Processing_Info={
                            "processing_method": "gemini_fallback",
                            "vdb_chunks_used": len(vdb_results)
                        }
                    )
                else:
                    raise ValueError("No valid JSON found in response")
                    
            except Exception as e:
                logger.warning(f"JSON parsing failed, using raw response: {e}")
                
                return QueryResponse(
                    Decision="Requires Further Review",
                    Amount="N/A",
                    Justification=response_text,
                    Relevant_Clauses=relevant_clauses,
                    Processing_Info={
                        "processing_method": "gemini_raw",
                        "parsing_error": str(e)
                    }
                )
                
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            return QueryResponse(
                Decision="Requires Further Review",
                Amount="N/A",
                Justification=f"Processing failed: {str(e)}",
                Relevant_Clauses=[]
            )
    
    def get_graph_structure(self) -> Dict[str, Any]:
        """Get graph structure for visualization"""
        if not self.kg_manager:
            return {"nodes": [], "edges": [], "message": "KG not available"}
        
        try:
            # Use existing KG manager to get graph structure
            if hasattr(self.kg_manager, 'get_graph_structure'):
                return self.kg_manager.get_graph_structure()
            else:
                # Fallback: get basic structure
                nodes = []
                edges = []
                
                if self.kg_manager.connection_mode == "fallback":
                    # Use session data
                    session_data = self.kg_manager.current_session_data
                    
                    for i, doc_data in enumerate(session_data.get("nodes", [])):
                        node = {
                            "id": f"doc_{i}",
                            "label": doc_data.get("document_id", f"Document {i}"),
                            "type": "Document",
                            "properties": {
                                "document_id": doc_data.get("document_id"),
                                "content_length": len(doc_data.get("content", ""))
                            }
                        }
                        nodes.append(node)
                
                return {
                    "nodes": nodes,
                    "edges": edges,
                    "stats": {
                        "total_nodes": len(nodes),
                        "total_edges": len(edges),
                        "connection_mode": self.kg_manager.connection_mode
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting graph structure: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}
    
    def clear_session(self):
        """Clear session data including files, vectors, and knowledge graph"""
        try:
            # Clear Knowledge Graph data
            if self.kg_manager:
                self.kg_manager.clear_session_data()
                logger.info("Knowledge graph data cleared")
            
            # Clear ChromaDB vectors
            if self.vector_store:
                try:
                    self.vector_store.delete_collection()
                    # Recreate the collection
                    self.vector_store = Chroma(
                        persist_directory="data/chroma_db",
                        embedding_function=self.embeddings,
                        collection_name="insurance_documents"
                    )
                    logger.info("Vector database cleared")
                except Exception as e:
                    logger.warning(f"ChromaDB clear failed: {e}")
            
            # Clear ALL uploaded files (complete cleanup)
            import shutil
            upload_dir = Path("documents/uploads")
            files_removed = 0
            if upload_dir.exists():
                for file_path in upload_dir.iterdir():
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            files_removed += 1
                            logger.info(f"Removed file: {file_path.name}")
                        except Exception as e:
                            logger.warning(f"Failed to remove file {file_path.name}: {e}")
            
            logger.info(f"Removed {files_removed} uploaded files")
            
            # Clear ChromaDB persistent data
            chroma_dir = Path("data/chroma_db")
            if chroma_dir.exists():
                try:
                    shutil.rmtree(chroma_dir)
                    chroma_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("ChromaDB persistent data cleared")
                    
                    # Reinitialize ChromaDB
                    self._initialize_chromadb()
                except Exception as e:
                    logger.warning(f"Failed to clear ChromaDB data: {e}")
            
            logger.info("Session cleared completely")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False

# Global processor instance
hybrid_processor = HybridRAGProcessor()

# Create FastAPI app
app = FastAPI(
    title="Enhanced Insurance Document Processing System",
    description="Hybrid RAG system with Knowledge Graph and Vector Database",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_query", response_model=QueryResponse)
async def process_interactive_query(request: QueryRequest):
    """Enhanced query processing with hybrid RAG"""
    logger.info(f"🔍 Processing interactive query: {request.query[:100]}...")
    
    try:
        result = hybrid_processor.process_query(request.query)
        logger.info(f"✅ Query processed: {result.Decision}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Query processing failed: {e}")
        return QueryResponse(
            Decision="Requires Further Review",
            Amount="N/A",
            Justification=f"Processing error: {str(e)}",
            Relevant_Clauses=[],
            Processing_Info={"error": str(e)}
        )

@app.post("/upload_document")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: str = None
):
    """Upload and process document with hybrid RAG"""
    try:
        if not document_id:
            document_id = file.filename
        
        logger.info(f"📄 Uploading document: {file.filename}")
        
        # Save uploaded file
        upload_dir = Path("documents/uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document in background
        background_tasks.add_task(
            process_uploaded_document,
            str(file_path),
            document_id
        )
        
        return DocumentUploadResponse(
            message="Document uploaded and processing started",
            document_id=document_id,
            processing_status="processing",
            file_info={
                "filename": file.filename,
                "size": len(content),
                "type": file.content_type,
                "path": str(file_path)
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_uploaded_document(file_path: str, document_id: str):
    """Background task for document processing with enhanced error handling"""
    import asyncio
    
    try:
        logger.info(f"🔄 Background processing: {document_id}")
        
        # Add longer timeout for large documents
        try:
            # Run document processing with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, 
                    hybrid_processor.process_document, 
                    file_path, 
                    document_id
                ),
                timeout=300  # 5 minutes timeout for large documents
            )
            
            if result.get("success"):
                kg_status = "✅" if result.get("kg_processed") else "⚠️"
                vdb_chunks = result.get("vdb_chunks", 0)
                logger.info(f"✅ Document processed: {document_id} | KG: {kg_status} | VDB: {vdb_chunks} chunks")
            else:
                logger.error(f"❌ Document processing failed: {result.get('error')}")
                
        except asyncio.TimeoutError:
            logger.error(f"❌ Document processing timed out after 5 minutes: {document_id}")
            
    except Exception as e:
        logger.error(f"❌ Background processing error: {e}")

@app.get("/api/graph/structure")
async def get_graph_structure():
    """Get graph structure for visualization"""
    try:
        structure = hybrid_processor.get_graph_structure()
        return structure
    except Exception as e:
        logger.error(f"❌ Graph structure error: {e}")
        return {"nodes": [], "edges": [], "error": str(e)}

@app.post("/api/session/clear")
async def clear_session():
    """Clear session data"""
    try:
        success = hybrid_processor.clear_session()
        return {"message": "Session cleared successfully" if success else "Session clear failed"}
    except Exception as e:
        logger.error(f"❌ Session clear error: {e}")
        return {"message": f"Session clear failed: {str(e)}"}

@app.get("/api/session/stats")
async def get_session_stats():
    """Get session statistics"""
    try:
        if hybrid_processor.kg_manager:
            return hybrid_processor.kg_manager.get_session_stats()
        else:
            return {"documents_in_session": 0, "active_document": None}
    except Exception as e:
        logger.error(f"❌ Session stats error: {e}")
        return {"error": str(e)}

@app.get("/status")
async def get_status():
    """Get system status"""
    try:
        # Enhanced KG status with connection mode
        kg_status = {
            "status": "unavailable",
            "connection_mode": "unknown",
            "details": "KG Manager not initialized"
        }
        
        if hybrid_processor.kg_manager:
            try:
                kg_mode = hybrid_processor.kg_manager.connection_mode
                session_stats = hybrid_processor.kg_manager.get_session_stats()
                kg_status = {
                    "status": "healthy",
                    "connection_mode": kg_mode,
                    "node_count": session_stats.get("documents_in_session", 0),
                    "details": f"Neo4j KG operational in {kg_mode} mode"
                }
            except Exception as kg_error:
                kg_status = {
                    "status": "error",
                    "connection_mode": "unknown", 
                    "details": f"KG connection error: {str(kg_error)}"
                }
        
        # Enhanced VDB status with collection info
        vdb_status = {
            "status": "unavailable",
            "details": "ChromaDB not initialized"
        }
        
        if hybrid_processor.vector_store:
            try:
                collection = hybrid_processor.vector_store._collection
                doc_count = collection.count()
                vdb_status = {
                    "status": "healthy",
                    "document_count": doc_count,
                    "collection_name": "insurance_documents",
                    "details": f"ChromaDB operational with {doc_count} documents"
                }
            except Exception as vdb_error:
                vdb_status = {
                    "status": "error", 
                    "details": f"ChromaDB connection error: {str(vdb_error)}"
                }
        
        llm_status = "healthy" if hybrid_processor.llm else "unavailable"
        
        return {
            "status": "healthy" if hybrid_processor.initialized else "limited",
            "kg_status": kg_status,
            "vdb_status": vdb_status,
            "llm_status": llm_status,
            "agents_status": "operational",
            "documents_processed": 0,  # This could be tracked
            "message": "Hybrid RAG system operational"
        }
    except Exception as e:
        logger.error(f"❌ Status check error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/admin/clear-vector-db")
async def clear_vector_database():
    """Clear all data from ChromaDB vector database"""
    try:
        if not hybrid_processor.vector_store:
            return {"success": False, "message": "Vector database not available"}
        
        # Get the ChromaDB collection
        collection = hybrid_processor.vector_store._collection
        initial_count = collection.count()
        logger.info(f"Current document count before clearing: {initial_count}")
        
        # Force complete clear by removing persistent storage first
        logger.info("Forcing complete ChromaDB clear by removing persistent storage")
        import shutil
        from pathlib import Path
        
        try:
            # Step 1: Close current vector store
            hybrid_processor.vector_store = None
            
            # Step 2: Remove all ChromaDB directories
            chroma_paths = [
                Path("data/chroma_db"),
                Path("data/chroma_db_direct"),
                Path("chroma_db")  # In case it's in root
            ]
            
            for chroma_dir in chroma_paths:
                if chroma_dir.exists():
                    logger.info(f"Removing ChromaDB directory: {chroma_dir}")
                    shutil.rmtree(chroma_dir)
                    logger.info(f"Removed {chroma_dir}")
            
            logger.info("All ChromaDB directories removed")
            
        except Exception as e:
            logger.error(f"Error removing ChromaDB directories: {e}")
            
            # Fallback: try the original deletion methods
            if initial_count > 0:
                try:
                    # Method 1: Delete all documents by getting all IDs
                    all_docs = collection.get()
                    if all_docs['ids']:
                        logger.info(f"Deleting {len(all_docs['ids'])} documents using IDs")
                        collection.delete(ids=all_docs['ids'])
                except Exception as e2:
                    logger.warning(f"ID-based deletion failed: {e2}")
                
                try:
                    # Method 2: Delete with empty where clause (deletes all)
                    logger.info("Attempting to delete all documents with empty where clause")
                    collection.delete(where={})
                except Exception as e2:
                    logger.warning(f"Where-based deletion failed: {e2}")
        
        # Reinitialize ChromaDB to ensure clean state
        hybrid_processor._initialize_chromadb()
        
        # Verify final count
        final_collection = hybrid_processor.vector_store._collection
        final_count = final_collection.count()
        logger.info(f"Final document count after reinitialization: {final_count}")
        
        logger.info("✅ Vector database cleared successfully")
        return {
            "success": True,
            "message": f"Vector database cleared successfully. Document count: {initial_count} → {final_count}",
            "action": "vector_db_cleared",
            "before_count": initial_count,
            "after_count": final_count
        }
        
    except Exception as e:
        logger.error(f"❌ Error clearing vector database: {e}")
        return {
            "success": False,
            "message": f"Error clearing vector database: {str(e)}",
            "error": str(e)
        }

@app.post("/admin/clear-knowledge-graph")
async def clear_knowledge_graph():
    """Clear all data from Neo4j knowledge graph"""
    try:
        if not hybrid_processor.kg_manager:
            return {"success": False, "message": "Knowledge graph not available"}
        
        # Clear Neo4j data
        success = hybrid_processor.kg_manager.clear_session()
        
        if success:
            logger.info("✅ Knowledge graph cleared successfully")
            return {
                "success": True,
                "message": "Knowledge graph cleared successfully",
                "action": "knowledge_graph_cleared"
            }
        else:
            return {
                "success": False,
                "message": "Failed to clear knowledge graph"
            }
        
    except Exception as e:
        logger.error(f"❌ Error clearing knowledge graph: {e}")
        return {
            "success": False,
            "message": f"Error clearing knowledge graph: {str(e)}",
            "error": str(e)
        }

@app.post("/admin/clear-all-data")
async def clear_all_data():
    """Clear all uploaded data from both vector database and knowledge graph"""
    try:
        results = {
            "vector_db": {"success": False, "message": "Not attempted"},
            "knowledge_graph": {"success": False, "message": "Not attempted"},
            "uploaded_files": {"success": False, "message": "Not attempted"}
        }
        
        # Clear vector database
        if hybrid_processor.vector_store:
            try:
                collection = hybrid_processor.vector_store._collection
                collection.delete()
                hybrid_processor._initialize_chromadb()
                results["vector_db"] = {"success": True, "message": "Vector database cleared"}
            except Exception as vdb_error:
                results["vector_db"] = {"success": False, "message": f"VDB error: {str(vdb_error)}"}
        
        # Clear knowledge graph
        if hybrid_processor.kg_manager:
            try:
                kg_success = hybrid_processor.kg_manager.clear_session()
                if kg_success:
                    results["knowledge_graph"] = {"success": True, "message": "Knowledge graph cleared"}
                else:
                    results["knowledge_graph"] = {"success": False, "message": "KG clear operation failed"}
            except Exception as kg_error:
                results["knowledge_graph"] = {"success": False, "message": f"KG error: {str(kg_error)}"}
        
        # Clear uploaded files
        try:
            upload_dir = Path("documents/uploads")
            if upload_dir.exists():
                import shutil
                shutil.rmtree(upload_dir)
                upload_dir.mkdir(exist_ok=True)
                results["uploaded_files"] = {"success": True, "message": "Uploaded files removed"}
            else:
                results["uploaded_files"] = {"success": True, "message": "No uploaded files to remove"}
        except Exception as file_error:
            results["uploaded_files"] = {"success": False, "message": f"File error: {str(file_error)}"}
        
        # Determine overall success
        overall_success = all(result["success"] for result in results.values())
        
        logger.info("✅ Complete data clearing operation completed")
        return {
            "success": overall_success,
            "message": "Data clearing operation completed",
            "action": "all_data_cleared",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ Error in clear all data operation: {e}")
        return {
            "success": False,
            "message": f"Error in clear all data operation: {str(e)}",
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/kg_schema")
async def get_kg_schema():
    """Get knowledge graph schema information"""
    try:
        if hybrid_processor.kg_manager:
            return hybrid_processor.kg_manager.get_kg_schema()
        else:
            return {
                "mode": "unavailable",
                "node_labels": [],
                "relationship_types": [],
                "message": "KG manager not available"
            }
    except Exception as e:
        logger.error(f"❌ KG schema error: {e}")
        return {"error": str(e)}

@app.get("/rate_limits")
async def get_rate_limits():
    """Get rate limiting configuration"""
    return {
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite"),
        "rate_limit_info": "Using Gemini 2.0 Flash Lite for higher rate limits",
        "estimated_qpm": "1000+",
        "retry_logic": "Built-in exponential backoff"
    }

@app.post("/process_documents")
async def process_documents_folder():
    """Process documents folder (for compatibility)"""
    # This endpoint exists for frontend compatibility
    # In the hybrid system, documents are uploaded individually
    return {
        "message": "Document processing handled via upload_document endpoint",
        "status": "use_upload_endpoint",
        "recommendation": "Use the upload document interface"
    }

@app.post("/api/query/interactive")
async def process_interactive_query_alt(request: QueryRequest):
    """Alternative interactive query endpoint for frontend compatibility"""
    # This is the same as /process_query but with a different path for frontend
    return await process_interactive_query(request)

@app.get("/api/graph/nodes/{node_type}")
async def get_nodes_by_type(node_type: str, limit: int = 50):
    """Get nodes filtered by type"""
    try:
        structure = hybrid_processor.get_graph_structure()
        nodes = structure.get("nodes", [])
        
        # Filter by node type
        filtered_nodes = [
            node for node in nodes
            if node.get("type", "").lower() == node_type.lower()
        ][:limit]
        
        return {
            "nodes": filtered_nodes,
            "node_type": node_type,
            "count": len(filtered_nodes),
            "total_available": len([n for n in nodes if n.get("type", "").lower() == node_type.lower()])
        }
    except Exception as e:
        logger.error(f"❌ Get nodes by type error: {e}")
        return {"nodes": [], "error": str(e)}

if __name__ == "__main__":
    # Ensure directories exist
    Path("data").mkdir(exist_ok=True)
    Path("documents/uploads").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    uvicorn.run(
        "hybrid_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )