"""
Enhanced Knowledge Graph Manager with PDF Support and Fallback Modes
Supports multiple Neo4j configurations and document processing
"""

import os
import logging
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError
import json
import re
# from document_processor import DocumentProcessor  # Not needed for hybrid system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedKnowledgeGraphManager:
    """
    Enhanced KG Manager with PDF support and multiple connection modes
    """
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j connection with enhanced configuration
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
        self.connection_mode = "unknown"
        self.current_session_data = {
            "nodes": [],
            "relationships": [],
            "queries": [],
            "active_document": None
        }
        self.fallback_data = self.current_session_data  # Alias for compatibility
        
        # Document processor not needed in hybrid system
        # Processing is handled by hybrid_main.py  
        self.doc_processor = None
        
    def initialize_kg(self) -> bool:
        """
        Initialize KG with multiple connection attempts and fallback modes
        """
        # Try Neo4j Aura connection
        if self._try_aura_connection():
            return True
            
        # Try local Neo4j connection
        if self._try_local_connection():
            return True
            
        # Use fallback mode
        logger.warning("Neo4j connection failed - using fallback mode")
        self.connection_mode = "fallback"
        return True  # Always return True to allow system to continue
    
    def _try_aura_connection(self) -> bool:
        """Try connecting to Neo4j Aura"""
        try:
            logger.info("Attempting Neo4j Aura connection...")
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password),
                database=self.database
            )
            
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info("Successfully connected to Neo4j Aura")
                    self.connection_mode = "aura"
                    self._create_schema()
                    return True
                    
        except Exception as e:
            logger.warning(f"Neo4j Aura connection failed: {e}")
            if self.driver:
                self.driver.close()
                self.driver = None
            return False
    
    def _try_local_connection(self) -> bool:
        """Try connecting to local Neo4j"""
        try:
            logger.info("Attempting local Neo4j connection...")
            local_uri = "bolt://localhost:7687"
            self.driver = GraphDatabase.driver(
                local_uri, 
                auth=(self.username, self.password)
            )
            
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info("Successfully connected to local Neo4j")
                    self.connection_mode = "local"
                    self._create_schema()
                    return True
                    
        except Exception as e:
            logger.warning(f"Local Neo4j connection failed: {e}")
            if self.driver:
                self.driver.close()
                self.driver = None
            return False
    
    def _create_schema(self):
        """Create basic schema constraints and indexes"""
        if self.connection_mode == "fallback":
            return
            
        schema_queries = [
            "CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (p:Policy) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT procedure_name IF NOT EXISTS FOR (pr:Procedure) REQUIRE pr.name IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE INDEX policy_type_idx IF NOT EXISTS FOR (p:Policy) ON (p.type)",
            "CREATE INDEX procedure_category_idx IF NOT EXISTS FOR (pr:Procedure) ON (pr.category)",
        ]
        
        with self.driver.session() as session:
            for query in schema_queries:
                try:
                    session.run(query)
                    logger.debug(f"Schema query executed: {query[:50]}...")
                except Exception as e:
                    logger.debug(f"Schema query warning: {e}")
    
    def process_document_to_kg(self, document_path: str, document_id: str = None) -> bool:
        """
        Process document (PDF or text) and add to Knowledge Graph
        """
        try:
            if not self.doc_processor:
                logger.error("Document processor not initialized")
                return False
            
            # Process document with AI
            result = self.doc_processor.process_document(document_path, document_id)
            
            if "error" in result:
                logger.error(f"Document processing failed: {result['error']}")
                return False
            
            # Store in Knowledge Graph or fallback
            return self._store_processed_document(result)
            
        except Exception as e:
            logger.error(f"Error processing document {document_path}: {e}")
            return False
    
    def _store_processed_document(self, doc_result: Dict[str, Any]) -> bool:
        """Store processed document in KG or fallback storage"""
        try:
            entities = doc_result.get("entities", {})
            document_id = doc_result.get("document_id")
            
            if self.connection_mode == "fallback":
                return self._store_in_fallback(doc_result)
            else:
                return self._store_in_neo4j(entities, document_id, doc_result.get("text_content", ""))
                
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            return False
    
    def _store_in_neo4j(self, entities: Dict, document_id: str, content: str) -> bool:
        """Store entities in Neo4j database"""
        try:
            with self.driver.session() as session:
                # Clear all existing data before storing new document
                logger.info("Clearing existing Neo4j data before storing new document")
                session.run("MATCH (n) DETACH DELETE n")
                
                # Store document node
                session.run(
                    "MERGE (d:Document {id: $doc_id}) "
                    "SET d.content = $content, d.processed_at = datetime(), "
                    "d.title = $title, d.type = $type",
                    doc_id=document_id,
                    content=content[:2000],  # Truncate for storage
                    title=entities.get("document_metadata", {}).get("title", "Unknown"),
                    type=entities.get("document_metadata", {}).get("type", "document")
                )
                
                # Store policies
                for policy in entities.get("policies", []):
                    session.run(
                        "MERGE (p:Policy {id: $id}) "
                        "SET p.name = $name, p.type = $type, "
                        "p.coverage_amount = $coverage_amount, p.premium = $premium, "
                        "p.term = $term "
                        "WITH p "
                        "MATCH (d:Document {id: $doc_id}) "
                        "MERGE (p)-[:DEFINED_IN]->(d)",
                        id=policy.get("id", f"policy_{len(entities.get('policies', []))}"),
                        name=policy.get("name", "Unknown Policy"),
                        type=policy.get("type", "insurance"),
                        coverage_amount=policy.get("coverage_amount", "0"),
                        premium=policy.get("premium", "0"),
                        term=policy.get("term", "1 year"),
                        doc_id=document_id
                    )
                
                # Store procedures
                for procedure in entities.get("procedures", []):
                    session.run(
                        "MERGE (pr:Procedure {name: $name}) "
                        "SET pr.category = $category, pr.coverage_limit = $coverage_limit, "
                        "pr.waiting_period = $waiting_period, pr.conditions = $conditions "
                        "WITH pr "
                        "MATCH (d:Document {id: $doc_id}) "
                        "MERGE (pr)-[:DEFINED_IN]->(d)",
                        name=procedure.get("name", "Unknown Procedure"),
                        category=procedure.get("category", "medical"),
                        coverage_limit=procedure.get("coverage_limit", "0"),
                        waiting_period=procedure.get("waiting_period", "0"),
                        conditions=procedure.get("conditions", ""),
                        doc_id=document_id
                    )
                
                # Store eligibility criteria
                for criteria in entities.get("eligibility_criteria", []):
                    session.run(
                        "MERGE (e:EligibilityCriteria {type: $type}) "
                        "SET e.min_value = $min_value, e.max_value = $max_value, "
                        "e.description = $description, e.conditions = $conditions "
                        "WITH e "
                        "MATCH (d:Document {id: $doc_id}) "
                        "MERGE (e)-[:DEFINED_IN]->(d)",
                        type=criteria.get("type", "unknown"),
                        min_value=criteria.get("min_value", ""),
                        max_value=criteria.get("max_value", ""),
                        description=criteria.get("description", ""),
                        conditions=criteria.get("conditions", ""),
                        doc_id=document_id
                    )
                
                # Create relationships between policies and procedures
                policy_ids = [p.get("id", f"policy_{i}") for i, p in enumerate(entities.get("policies", []))]
                procedure_names = [p.get("name") for p in entities.get("procedures", [])]
                
                for policy_id in policy_ids:
                    for proc_name in procedure_names:
                        session.run(
                            "MATCH (p:Policy {id: $policy_id}) "
                            "MATCH (pr:Procedure {name: $proc_name}) "
                            "MERGE (p)-[r:COVERS_PROCEDURE]->(pr)",
                            policy_id=policy_id,
                            proc_name=proc_name
                        )
                
                logger.info(f"Successfully stored document {document_id} in Neo4j")
                return True
                
        except Exception as e:
            logger.error(f"Error storing in Neo4j: {e}")
            return False
    
    def _store_in_fallback(self, doc_result: Dict[str, Any]) -> bool:
        """Store document in fallback memory storage (session-based)"""
        try:
            entities = doc_result.get("entities", {})
            document_id = doc_result.get("document_id")
            
            # Clear previous session data before storing new document
            self.clear_session_data()
            
            # Store as searchable data structure
            doc_data = {
                "document_id": document_id,
                "entities": entities,
                "content": doc_result.get("text_content", ""),
                "processed_at": "now"
            }
            
            self.current_session_data["nodes"].append(doc_data)
            self.current_session_data["active_document"] = document_id
            logger.info(f"Stored document {document_id} in session storage")
            return True
            
        except Exception as e:
            logger.error(f"Error storing in fallback: {e}")
            return False
    
    def query_kg(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Execute Cypher query or fallback search
        """
        if self.connection_mode == "fallback":
            return self._fallback_query(cypher_query)
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = []
                for record in result:
                    records.append(dict(record))
                return records
                
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return self._fallback_query(cypher_query)
    
    def _fallback_query(self, query_intent: str) -> List[Dict[str, Any]]:
        """Fallback query using in-memory data"""
        results = []
        
        # Simple keyword matching in fallback data
        keywords = re.findall(r'\b\w+\b', query_intent.lower())
        
        for node_data in self.fallback_data["nodes"]:
            content = str(node_data).lower()
            matches = sum(1 for keyword in keywords if keyword in content)
            
            if matches > 0:
                results.append({
                    "document_id": node_data.get("document_id"),
                    "match_score": matches,
                    "entities": node_data.get("entities", {}),
                    "source": "fallback_storage"
                })
        
        # Sort by match score
        results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        return results[:10]  # Return top 10 matches
    
    def get_kg_schema(self) -> Dict[str, Any]:
        """Get KG schema information"""
        if self.connection_mode == "fallback":
            return {
                "mode": "fallback",
                "node_labels": ["Document", "Policy", "Procedure", "EligibilityCriteria"],
                "relationship_types": ["DEFINED_IN", "COVERS_PROCEDURE"],
                "documents_stored": len(self.fallback_data["nodes"])
            }
        
        schema_queries = {
            "node_labels": "CALL db.labels()",
            "relationship_types": "CALL db.relationshipTypes()",
            "property_keys": "CALL db.propertyKeys()"
        }
        
        schema_info = {"mode": self.connection_mode}
        for key, query in schema_queries.items():
            try:
                result = self.query_kg(query)
                if key == "node_labels":
                    schema_info[key] = [record.get("label", "") for record in result]
                elif key == "relationship_types":
                    schema_info[key] = [record.get("relationshipType", "") for record in result]
                else:
                    schema_info[key] = [record.get("propertyKey", "") for record in result]
            except Exception as e:
                logger.error(f"Error getting {key}: {e}")
                schema_info[key] = []
        
        return schema_info
    
    def get_graph_structure(self) -> Dict[str, Any]:
        """Get the complete graph structure for visualization"""
        try:
            if self.connection_mode == "fallback":
                # Use session data for fallback mode
                return {
                    "nodes": self.current_session_data.get("nodes", []),
                    "edges": self.current_session_data.get("relationships", []),
                    "stats": {
                        "total_nodes": len(self.current_session_data.get("nodes", [])),
                        "total_edges": len(self.current_session_data.get("relationships", [])),
                        "connection_mode": self.connection_mode
                    }
                }
            
            # For Neo4j connection, fetch all nodes and relationships
            nodes = []
            edges = []
            
            # Get all nodes with their properties
            nodes_query = """
            MATCH (n)
            RETURN n.id as id, labels(n)[0] as type, n.name as label, properties(n) as properties
            LIMIT 1000
            """
            
            try:
                node_results = self.query_kg(nodes_query)
                for record in node_results:
                    node = {
                        "id": record.get("id", f"node_{len(nodes)}"),
                        "type": record.get("type", "Unknown"),
                        "label": record.get("label", record.get("id", "Unnamed")),
                        "properties": record.get("properties", {})
                    }
                    nodes.append(node)
                    
                logger.info(f"Fetched {len(nodes)} nodes from Neo4j")
                
            except Exception as e:
                logger.error(f"Error fetching nodes: {e}")
            
            # Get all relationships
            edges_query = """
            MATCH (a)-[r]->(b)
            RETURN a.id as source, b.id as target, type(r) as type, properties(r) as properties
            LIMIT 1000
            """
            
            try:
                edge_results = self.query_kg(edges_query)
                for record in edge_results:
                    edge = {
                        "source": record.get("source"),
                        "target": record.get("target"), 
                        "type": record.get("type", "RELATED_TO"),
                        "properties": record.get("properties", {})
                    }
                    edges.append(edge)
                    
                logger.info(f"Fetched {len(edges)} relationships from Neo4j")
                
            except Exception as e:
                logger.error(f"Error fetching relationships: {e}")
            
            return {
                "nodes": nodes,
                "edges": edges,
                "stats": {
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "connection_mode": self.connection_mode
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
                    "connection_mode": self.connection_mode,
                    "error": str(e)
                }
            }
    
    def process_documents_folder(self, folder_path: str) -> Dict[str, Any]:
        """Process all documents in a folder"""
        if not self.doc_processor:
            return {"error": "Document processor not initialized"}
        
        results = self.doc_processor.process_documents_folder(folder_path)
        
        # Store each processed document
        stored_count = 0
        for result in results:
            if "error" not in result:
                if self._store_processed_document(result):
                    stored_count += 1
        
        return {
            "processed_documents": len(results),
            "stored_documents": stored_count,
            "connection_mode": self.connection_mode,
            "results": results
        }
    
    def clear_session_data(self):
        """Clear current session data"""
        logger.info("Clearing session data")
        self.current_session_data = {
            "nodes": [],
            "relationships": [],
            "queries": [],
            "active_document": None
        }
        self.fallback_data = self.current_session_data
        
        # Also clear Neo4j data if connected
        if self.connection_mode != "fallback" and self.driver:
            try:
                with self.driver.session() as session:
                    # Clear all nodes and relationships
                    session.run("MATCH (n) DETACH DELETE n")
                logger.info("Cleared Neo4j database")
            except Exception as e:
                logger.error(f"Error clearing Neo4j: {e}")
                return False
        return True
    
    def clear_session(self):
        """Alias for clear_session_data for API compatibility"""
        return self.clear_session_data()
    
    def get_session_stats(self):
        """Get current session statistics"""
        # Count actual document nodes only
        document_nodes = [node for node in self.current_session_data.get("nodes", []) 
                         if node.get("type") == "Document"]
        
        # Also check uploaded files directory
        from pathlib import Path
        upload_dir = Path("documents/uploads")
        uploaded_files = []
        if upload_dir.exists():
            uploaded_files = [f.name for f in upload_dir.iterdir() if f.is_file()]
        
        return {
            "documents_in_session": len(document_nodes),
            "uploaded_files": len(uploaded_files),
            "uploaded_file_names": uploaded_files,
            "active_document": self.current_session_data.get("active_document"),
            "connection_mode": self.connection_mode,
            "total_nodes": len(self.current_session_data.get("nodes", [])),
            "total_relationships": len(self.current_session_data.get("relationships", []))
        }
    
    def close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")


def create_enhanced_kg_manager() -> EnhancedKnowledgeGraphManager:
    """Factory function to create enhanced KG manager"""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    
    return EnhancedKnowledgeGraphManager(uri, username, password, database)