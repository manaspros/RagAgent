"""
Enhanced FastAPI Application with PDF Processing and Neo4j Integration
Supports PDF document analysis and Knowledge Graph-based reasoning
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

from kg_manager import create_enhanced_kg_manager, EnhancedKnowledgeGraphManager
from agents import create_multi_agent_orchestrator, MultiAgentOrchestrator
from rate_limiting_config import RateLimitConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Global variables
kg_manager: EnhancedKnowledgeGraphManager = None
orchestrator: MultiAgentOrchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan handler with PDF processing"""
    global kg_manager, orchestrator
    
    logger.info("Starting Enhanced LLM Document Processing System...")
    
    try:
        # Initialize Enhanced Knowledge Graph Manager
        kg_manager = create_enhanced_kg_manager()
        success = kg_manager.initialize_kg()
        
        if success:
            logger.info(f"Knowledge Graph initialized successfully in {kg_manager.connection_mode} mode")
        else:
            logger.warning("Knowledge Graph initialization had issues, but system will continue")
        
        # Initialize Multi-Agent Orchestrator (with fallback support)
        orchestrator = create_multi_agent_orchestrator(kg_manager)
        logger.info("Multi-Agent Orchestrator initialized successfully")
        
        # Skip processing existing documents - wait for user uploads
        logger.info("System ready for document uploads")
        
        logger.info("Enhanced system startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize enhanced system: {e}")
        # Don't raise - allow system to continue with limited functionality
        
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced LLM Document Processing System...")
    if kg_manager:
        kg_manager.close()
    logger.info("Enhanced system shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="Enhanced LLM Document Processing System",
    description="Multi-Agent system with PDF processing and Knowledge Graph integration",
    version="2.0.0-enhanced",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about insurance policy")

class ClauseReference(BaseModel):
    clause_text: str = Field(..., description="Text of the relevant policy clause")
    document_id: str = Field(..., description="Source document identifier")
    page_section: str = Field(..., description="Page or section reference")

class QueryResponse(BaseModel):
    Decision: str = Field(..., description="Approved, Rejected, or Requires Further Review")
    Amount: Any = Field(..., description="Payout amount or N/A")
    Justification: str = Field(..., description="Detailed explanation of the decision")
    Relevant_Clauses: list[ClauseReference] = Field(..., description="Relevant policy clauses")
    Processing_Info: Dict[str, Any] = Field(default={}, description="Additional processing information")

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str
    processing_status: str
    file_info: Dict[str, Any]

class SystemStatusResponse(BaseModel):
    status: str
    kg_status: str
    kg_mode: str
    agents_status: str
    documents_processed: int
    message: str


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with enhanced system information"""
    return {
        "message": "Enhanced LLM Document Processing System",
        "description": "Multi-Agent system with PDF processing and Knowledge Graph",
        "version": "2.0.0-enhanced",
        "features": [
            "PDF document processing",
            "Knowledge Graph integration",
            "Multi-agent reasoning",
            "Fallback modes for reliability"
        ],
        "endpoints": {
            "process_query": "POST /process_query - Process natural language queries",
            "upload_document": "POST /upload_document - Upload and process documents",
            "process_documents": "POST /process_documents - Process documents folder",
            "status": "GET /status - System status information",
            "kg_schema": "GET /kg_schema - Knowledge Graph schema",
            "rate_limits": "GET /rate_limits - Rate limiting configuration"
        }
    }

@app.post("/process_query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Enhanced query processing with Knowledge Graph integration"""
    try:
        logger.info(f"Processing enhanced query: {request.query}")
        
        if not orchestrator:
            # Fallback to direct Gemini processing
            return await fallback_query_processing(request.query)
        
        # Process through multi-agent system
        result = orchestrator.process_multi_agent_query(request.query)
        
        if "error" in result:
            logger.error(f"Multi-agent processing error: {result['error']}")
            return await fallback_query_processing(request.query)
        
        # Add processing information
        result["Processing_Info"] = {
            "kg_mode": kg_manager.connection_mode if kg_manager else "unknown",
            "processing_method": "multi_agent_system",
            "documents_available": len(kg_manager.fallback_data["nodes"]) if kg_manager else 0
        }
        
        # Format response
        formatted_result = format_query_response(result)
        logger.info(f"Enhanced query processed successfully: {formatted_result['Decision']}")
        return QueryResponse(**formatted_result)
        
    except Exception as e:
        logger.error(f"Error in enhanced query processing: {e}")
        return await fallback_query_processing(request.query)

async def fallback_query_processing(query: str) -> QueryResponse:
    """Fallback query processing using direct Gemini API"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Get available document context
        doc_context = ""
        if kg_manager and kg_manager.fallback_data["nodes"]:
            doc_context = f"\nAvailable documents: {len(kg_manager.fallback_data['nodes'])} processed documents with insurance policy information."
        
        prompt = f"""
        You are an enhanced insurance policy processing system with access to processed documents.
        
        Query: "{query}"
        {doc_context}
        
        Policy Context (Enhanced):
        - Multiple insurance policies may be available in the knowledge base
        - Standard coverage includes various medical procedures
        - Age eligibility, waiting periods, and geographic coverage apply
        - Enhanced analysis considers document-specific terms and conditions
        
        Provide a comprehensive analysis and return JSON in this format:
        {{
            "Decision": "Approved" | "Rejected" | "Requires Further Review",
            "Amount": number or "N/A",
            "Justification": "detailed analysis with enhanced reasoning",
            "Relevant_Clauses": [
                {{
                    "clause_text": "specific clause from processed documents",
                    "document_id": "source document",
                    "page_section": "section reference"
                }}
            ]
        }}
        """
        
        response = model.generate_content(prompt)
        result = parse_gemini_response(response.text)
        
        result["Processing_Info"] = {
            "processing_method": "fallback_gemini",
            "kg_mode": kg_manager.connection_mode if kg_manager else "unavailable"
        }
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Fallback processing error: {e}")
        return QueryResponse(
            Decision="Requires Further Review",
            Amount="N/A",
            Justification=f"System error during processing: {str(e)}",
            Relevant_Clauses=[],
            Processing_Info={"error": str(e)}
        )

@app.post("/upload_document")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: str = None
):
    """Upload and process a document (PDF or text)"""
    try:
        if not kg_manager:
            raise HTTPException(status_code=503, detail="Knowledge Graph manager not available")
        
        # Validate file type
        allowed_extensions = {'.pdf', '.txt', '.md'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Allowed: {allowed_extensions}"
            )
        
        # Save uploaded file
        upload_dir = Path("documents/uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document in background
        if not document_id:
            document_id = file.filename
        
        background_tasks.add_task(
            process_uploaded_document,
            str(file_path),
            document_id
        )
        
        return DocumentUploadResponse(
            message=f"Document upload successful. Processing started.",
            document_id=document_id,
            processing_status="processing",
            file_info={
                "filename": file.filename,
                "size": len(content),
                "type": file_extension,
                "path": str(file_path)
            }
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_documents")
async def process_documents_folder(background_tasks: BackgroundTasks):
    """Process all documents in the documents folder"""
    try:
        if not kg_manager:
            raise HTTPException(status_code=503, detail="Knowledge Graph manager not available")
        
        documents_dir = "documents"
        
        background_tasks.add_task(process_documents_background, documents_dir)
        
        return {
            "message": "Document processing started for entire documents folder",
            "status": "processing",
            "folder": documents_dir
        }
        
    except Exception as e:
        logger.error(f"Error processing documents folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=SystemStatusResponse)
async def get_enhanced_status():
    """Get enhanced system status"""
    try:
        kg_status = "connected" if kg_manager and kg_manager.driver else "fallback"
        kg_mode = kg_manager.connection_mode if kg_manager else "unknown"
        agents_status = "initialized" if orchestrator else "not_initialized"
        
        documents_processed = 0
        if kg_manager:
            if kg_manager.connection_mode == "fallback":
                documents_processed = len(kg_manager.fallback_data["nodes"])
            else:
                # Could query Neo4j for document count
                documents_processed = 0
        
        overall_status = "healthy" if kg_manager and orchestrator else "limited"
        
        return SystemStatusResponse(
            status=overall_status,
            kg_status=kg_status,
            kg_mode=kg_mode,
            agents_status=agents_status,
            documents_processed=documents_processed,
            message=f"Enhanced system is {overall_status}. KG: {kg_mode}, Documents: {documents_processed}"
        )
        
    except Exception as e:
        logger.error(f"Error getting enhanced status: {e}")
        return SystemStatusResponse(
            status="error",
            kg_status="error",
            kg_mode="error",
            agents_status="error",
            documents_processed=0,
            message=f"Error retrieving system status: {str(e)}"
        )

@app.get("/kg_schema")
async def get_kg_schema():
    """Get Knowledge Graph schema information"""
    try:
        if not kg_manager:
            raise HTTPException(status_code=503, detail="Knowledge Graph manager not available")
        
        schema_info = kg_manager.get_kg_schema()
        return {
            "message": "Knowledge Graph schema information",
            "schema": schema_info
        }
        
    except Exception as e:
        logger.error(f"Error getting KG schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rate_limits") 
async def get_rate_limits():
    """Get current rate limiting configuration"""
    try:
        return {
            "message": "Current rate limiting configuration",
            "config": RateLimitConfig.get_rate_limit_info(),
            "notes": {
                "delays": "All delays are in seconds",
                "purpose": "Prevents hitting Gemini API rate limits",
                "customization": "Set environment variables to customize delays"
            }
        }
    except Exception as e:
        logger.error(f"Error getting rate limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "message": "Enhanced LLM Document Processing System is running",
        "features": {
            "pdf_processing": "enabled",
            "kg_integration": kg_manager.connection_mode if kg_manager else "disabled",
            "gemini_api": "configured" if os.getenv('GEMINI_API_KEY') else "not_configured",
            "rate_limiting": "enabled"
        }
    }

@app.get("/api/graph/structure")
async def get_graph_structure():
    """Get complete graph topology for visualization"""
    try:
        if not kg_manager:
            raise HTTPException(status_code=503, detail="Knowledge Graph manager not available")
        
        if kg_manager.connection_mode == "fallback":
            # Use fallback data
            nodes = []
            edges = []
            node_id_counter = 1
            
            for doc_data in kg_manager.fallback_data["nodes"]:
                doc_id = f"doc_{node_id_counter}"
                entities = doc_data.get("entities", {})
                
                # Add document node
                nodes.append({
                    "id": doc_id,
                    "label": entities.get("document_metadata", {}).get("title", doc_data.get("document_id", "Document")),
                    "type": "Document",
                    "properties": {
                        "document_id": doc_data.get("document_id"),
                        "processed_at": doc_data.get("processed_at"),
                        "content_length": len(doc_data.get("content", ""))
                    }
                })
                
                # Add policy nodes
                for i, policy in enumerate(entities.get("policies", [])):
                    policy_id = f"policy_{node_id_counter}_{i}"
                    nodes.append({
                        "id": policy_id,
                        "label": policy.get("name", f"Policy {i+1}"),
                        "type": "Policy",
                        "properties": policy
                    })
                    edges.append({
                        "id": f"{doc_id}-{policy_id}",
                        "source": doc_id,
                        "target": policy_id,
                        "type": "DEFINED_IN",
                        "properties": {}
                    })
                
                # Add procedure nodes
                for i, procedure in enumerate(entities.get("procedures", [])):
                    proc_id = f"proc_{node_id_counter}_{i}"
                    nodes.append({
                        "id": proc_id,
                        "label": procedure.get("name", f"Procedure {i+1}"),
                        "type": "Procedure",
                        "properties": procedure
                    })
                    edges.append({
                        "id": f"{doc_id}-{proc_id}",
                        "source": doc_id,
                        "target": proc_id,
                        "type": "DEFINED_IN",
                        "properties": {}
                    })
                
                # Add eligibility criteria nodes
                for i, criteria in enumerate(entities.get("eligibility_criteria", [])):
                    crit_id = f"crit_{node_id_counter}_{i}"
                    nodes.append({
                        "id": crit_id,
                        "label": criteria.get("type", f"Criteria {i+1}"),
                        "type": "EligibilityCriteria",
                        "properties": criteria
                    })
                    edges.append({
                        "id": f"{doc_id}-{crit_id}",
                        "source": doc_id,
                        "target": crit_id,
                        "type": "DEFINED_IN",
                        "properties": {}
                    })
                
                # Add location nodes
                for i, location in enumerate(entities.get("geographic_coverage", [])):
                    loc_id = f"loc_{node_id_counter}_{i}"
                    nodes.append({
                        "id": loc_id,
                        "label": location.get("location", f"Location {i+1}"),
                        "type": "Location",
                        "properties": location
                    })
                    edges.append({
                        "id": f"{doc_id}-{loc_id}",
                        "source": doc_id,
                        "target": loc_id,
                        "type": "COVERS_LOCATION",
                        "properties": {}
                    })
                
                node_id_counter += 1
            
        else:
            # Use Neo4j data
            nodes_query = "MATCH (n) RETURN n, labels(n) as labels, id(n) as node_id LIMIT 100"
            edges_query = "MATCH (a)-[r]->(b) RETURN a, r, b, type(r) as rel_type, id(a) as source_id, id(b) as target_id LIMIT 200"
            
            nodes_result = kg_manager.query_kg(nodes_query)
            edges_result = kg_manager.query_kg(edges_query)
            
            # Transform to frontend format
            nodes = []
            for record in nodes_result:
                node_data = record.get('n', {})
                labels = record.get('labels', [])
                node_id = str(record.get('node_id', ''))
                
                nodes.append({
                    "id": node_id,
                    "label": node_data.get('name', node_data.get('title', f"Node {node_id}")),
                    "type": labels[0] if labels else "Unknown",
                    "properties": dict(node_data)
                })
            
            edges = []
            for record in edges_result:
                rel_data = record.get('r', {})
                rel_type = record.get('rel_type', 'RELATED')
                source_id = str(record.get('source_id', ''))
                target_id = str(record.get('target_id', ''))
                
                edges.append({
                    "id": f"{source_id}-{target_id}",
                    "source": source_id,
                    "target": target_id,
                    "type": rel_type,
                    "properties": dict(rel_data)
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "node_types": list(set(node["type"] for node in nodes)) if nodes else [],
                "edge_types": list(set(edge["type"] for edge in edges)) if edges else [],
                "connection_mode": kg_manager.connection_mode
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting graph structure: {e}")
        return {
            "nodes": [],
            "edges": [],
            "stats": {
                "total_nodes": 0,
                "total_edges": 0,
                "node_types": [],
                "edge_types": [],
                "connection_mode": "error",
                "error": str(e)
            }
        }

@app.get("/api/graph/nodes/{node_type}")
async def get_nodes_by_type(node_type: str):
    """Get filtered node data by type"""
    try:
        if not kg_manager:
            raise HTTPException(status_code=503, detail="Knowledge Graph manager not available")
        
        query = f"MATCH (n:{node_type}) RETURN n, id(n) as node_id LIMIT 50"
        result = kg_manager.query_kg(query)
        
        nodes = []
        for record in result:
            node_data = record.get('n', {})
            node_id = str(record.get('node_id', ''))
            
            nodes.append({
                "id": node_id,
                "label": node_data.get('name', node_data.get('title', f"{node_type} {node_id}")),
                "type": node_type,
                "properties": dict(node_data)
            })
        
        return {"nodes": nodes, "count": len(nodes)}
        
    except Exception as e:
        logger.error(f"Error getting nodes by type {node_type}: {e}")
        return {"nodes": [], "count": 0, "error": str(e)}

@app.post("/api/query/interactive")
async def interactive_query(request: QueryRequest):
    """Enhanced query with graph data"""
    try:
        # Process query through multi-agent system
        result = await process_query(request)
        
        # Add graph visualization data if available
        if kg_manager and hasattr(result, 'dict'):
            result_dict = result.dict() if hasattr(result, 'dict') else result
            
            # Get related graph data based on query
            try:
                graph_query = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 20"
                graph_result = kg_manager.query_kg(graph_query)
                
                result_dict["graph_data"] = {
                    "related_nodes": len(graph_result),
                    "visualization_available": True
                }
            except:
                result_dict["graph_data"] = {
                    "related_nodes": 0,
                    "visualization_available": False
                }
            
            return result_dict
        
        return result
        
    except Exception as e:
        logger.error(f"Error in interactive query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def process_uploaded_document(file_path: str, document_id: str):
    """Background task for processing uploaded document"""
    try:
        logger.info(f"Background processing of uploaded document: {file_path}")
        success = kg_manager.process_document_to_kg(file_path, document_id)
        
        if success:
            logger.info(f"Successfully processed uploaded document: {document_id}")
        else:
            logger.error(f"Failed to process uploaded document: {document_id}")
            
    except Exception as e:
        logger.error(f"Error in background document processing: {e}")

async def process_documents_background(folder_path: str):
    """Background task for processing documents folder"""
    try:
        logger.info(f"Background processing of documents folder: {folder_path}")
        result = kg_manager.process_documents_folder(folder_path)
        
        logger.info(f"Folder processing complete: {result}")
        
    except Exception as e:
        logger.error(f"Error in background folder processing: {e}")

async def process_existing_documents():
    """Skip processing existing documents - user will upload manually"""
    logger.info("Skipping existing document processing - waiting for user uploads")

# Helper functions
def format_query_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format query response to match expected structure"""
    formatted = {
        "Decision": result.get("Decision", "Requires Further Review"),
        "Amount": result.get("Amount", "N/A"),
        "Justification": result.get("Justification", "Processing completed with available information"),
        "Relevant_Clauses": [],
        "Processing_Info": result.get("Processing_Info", {})
    }
    
    # Format clauses
    clauses = result.get("Relevant_Clauses", [])
    for clause in clauses:
        if isinstance(clause, dict):
            formatted["Relevant_Clauses"].append(ClauseReference(
                clause_text=clause.get("clause_text", "Policy terms apply"),
                document_id=clause.get("document_id", "processed_document"),
                page_section=clause.get("page_section", "General Terms")
            ))
    
    return formatted

def parse_gemini_response(response_text: str) -> Dict[str, Any]:
    """Parse Gemini API response and extract JSON"""
    import json
    import re
    
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    # Fallback response
    return {
        "Decision": "Requires Further Review",
        "Amount": "N/A",
        "Justification": "Unable to parse AI response. Manual review recommended.",
        "Relevant_Clauses": [
            {
                "clause_text": "System processing limitations",
                "document_id": "system_response",
                "page_section": "Error Handling"
            }
        ]
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    
    # Check required environment variables
    required_vars = ["GEMINI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set these environment variables before starting the server")
        exit(1)
    
    logger.info(f"Starting enhanced server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )