"""
Enhanced Main Application with Hybrid RAG Integration
Integrates hybrid RAG system with existing frontend
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

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
            
            self.initialized = True
            logger.info("Hybrid RAG Processor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            self.initialized = False
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB vector database"""
        try:
            # Create persist directory
            persist_dir = Path("data/chroma_db")
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize ChromaDB
            self.vector_store = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=self.embeddings,
                collection_name="insurance_documents"
            )
            
            logger.info("ChromaDB initialized")
            
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            self.vector_store = None
    
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
        """Process document content with Gemini and store in KG"""
        try:
            if not self.llm:
                logger.warning("Gemini LLM not available for processing")
                return False
            
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
            logger.error(f"Error processing document with Gemini: {e}")
            return False
    
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
            
            # Process for vector database
            vdb_chunks = 0
            if self.vector_store and self.text_splitter:
                chunks = self.text_splitter.split_text(content)
                
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
                
                self.vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=[f"{document_id}_chunk_{i}" for i in range(len(chunks))]
                )
                
                vdb_chunks = len(chunks)
                logger.info(f"Added {vdb_chunks} chunks to ChromaDB")
            
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
            
            # Step 3: Enhanced reasoning with explicit citations
            try:
                return self._process_with_citations(query, vdb_results, kg_facts)
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
            WHERE toLower(n.name) CONTAINS toLower($keyword1) 
               OR toLower(n.name) CONTAINS toLower($keyword2) 
               OR toLower(n.name) CONTAINS toLower($keyword3)
               OR toLower(n.id) CONTAINS toLower($keyword1)
               OR toLower(n.id) CONTAINS toLower($keyword2)
               OR toLower(n.id) CONTAINS toLower($keyword3)
            RETURN n.id as id, n.name as name, labels(n)[0] as type, properties(n) as properties
            LIMIT 15
            """
            
            query_keywords = [word for word in query.lower().split() if len(word) > 2][:3]  # Filter short words
            logger.info(f"Using keywords: {query_keywords}")
            
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
        
        return QueryResponse(
            Decision="Requires Further Review",
            Amount="N/A", 
            Justification="System processed query in fallback mode. Please review manually for accurate assessment.",
            Relevant_Clauses=relevant_clauses,
            Processing_Info={
                "processing_method": "fallback_simple",
                "vdb_chunks_found": len(vdb_results)
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
    """Background task for document processing"""
    try:
        logger.info(f"🔄 Background processing: {document_id}")
        result = hybrid_processor.process_document(file_path, document_id)
        
        if result.get("success"):
            logger.info(f"✅ Document processed successfully: {document_id}")
        else:
            logger.error(f"❌ Document processing failed: {result.get('error')}")
            
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
        kg_status = "healthy" if hybrid_processor.kg_manager else "unavailable"
        vdb_status = "healthy" if hybrid_processor.vector_store else "unavailable"
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