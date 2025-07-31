# 🚀 Enhanced Hybrid RAG Integration

**Complete integration of advanced Hybrid RAG system with existing frontend**

## 🌟 What's New

### ✨ **Enhanced Backend**
- **Google Gemini 2.0 Flash Lite**: Higher rate limits, better performance
- **ChromaDB Vector Database**: Semantic search over document chunks  
- **Hybrid RAG Architecture**: Combines Knowledge Graph + Vector Search
- **Multi-Agent Processing**: Specialized agents for parsing, reasoning, calculation
- **No Docker Required**: Runs locally with simple setup

### 🎯 **Frontend Integration**
- **Existing React app works unchanged**
- **Enhanced document processing** with hybrid RAG
- **Improved query responses** with vector + graph search
- **Interactive graph visualization** with more depth
- **Session-based processing** with automatic cleanup

## 🚀 Quick Start

### 1. **Setup Enhanced System**
```bash
# Install enhanced dependencies
pip install -r hybrid_requirements.txt

# Run setup script
python setup_hybrid_system.py
```

### 2. **Configure Environment**
```bash
# Edit .env file
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-lite
```

### 3. **Start System**
```bash
# Start enhanced backend
python hybrid_main.py

# Start existing frontend (in another terminal)
cd frontend
npm run dev
```

### 4. **Access Applications**
- **Frontend**: http://localhost:3000 (React app)
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🏗️ System Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   React Frontend    │    │  Enhanced Backend   │    │    Data Stores      │
│                     │    │                     │    │                     │
│ • Document Upload   │◄──►│ • Hybrid RAG        │◄──►│ • ChromaDB (VDB)    │
│ • Query Interface   │    │ • Gemini Agents     │    │ • Neo4j (KG)        │
│ • Graph Visualize   │    │ • Multi-Agent Sys   │    │ • Session Storage   │
│ • Session Mgmt      │    │ • FastAPI Server    │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## 🔄 Enhanced Workflow

### **Document Processing**
1. **Upload PDF** via existing frontend interface
2. **Hybrid Processing**:
   - Text extraction and chunking
   - ChromaDB vector embedding  
   - Knowledge graph entity extraction
   - Neo4j relationship mapping
3. **Session Management**: Previous data automatically cleared
4. **Real-time Updates**: Processing status via existing UI

### **Query Processing** 
1. **Enhanced Pipeline**:
   - Query parsing with Gemini
   - Vector similarity search (ChromaDB)
   - Knowledge graph traversal (Neo4j)
   - Policy reasoning with multi-agents
   - Financial calculations
   - Decision synthesis
2. **Improved Responses**:
   - More accurate decisions
   - Better source attribution
   - Comprehensive justifications
   - Graph data for visualization

### **Graph Visualization**
1. **Enhanced Depth**:
   - All document chunks visible
   - Entity relationships mapped
   - Source traceability maintained
   - Interactive exploration
2. **Real-time Updates**: Graph updates as documents processed

## 📊 Enhanced Features

### 🤖 **Multi-Agent System**
```python
# Query Processing Pipeline
QueryParsingAgent → HybridRetrievalAgent → PolicyReasoningAgent → 
FinancialCalculationAgent → DecisionSynthesisAgent
```

### 🧠 **Hybrid RAG**
- **Vector Search**: Semantic similarity over document chunks
- **Graph Search**: Explicit relationships and policy rules
- **Result Fusion**: LLM-powered combination of both sources
- **Source Attribution**: Complete traceability to original content

### ⚡ **Gemini Integration**
- **Model**: gemini-2.0-flash-lite (higher rate limits)
- **Chain-of-Thought**: Step-by-step reasoning
- **Function Calling**: Mathematical calculations
- **Retry Logic**: Robust error handling

## 🛠️ Technical Details

### **API Endpoints** (Fully Compatible)
```bash
# Document Management
POST /upload_document              # Enhanced with hybrid processing
POST /api/session/clear           # Session cleanup
GET  /api/session/stats           # Session statistics

# Query Processing  
POST /process_query               # Enhanced with hybrid RAG
POST /api/query/interactive       # Alternative endpoint

# Graph Visualization
GET  /api/graph/structure         # Enhanced graph data
GET  /api/graph/nodes/{type}      # Filtered node data

# System Information
GET  /status                      # Enhanced system status
GET  /health                      # Health check
GET  /kg_schema                   # KG schema info
GET  /rate_limits                 # Rate limit info
```

### **Data Flow**
```
PDF Upload → Text Extraction → Chunking → Vector Embedding → ChromaDB
                ↓
           Entity Extraction → Relationship Mapping → Neo4j KG
                ↓
           Session Data → Frontend Updates → Graph Visualization
```

### **Enhanced Response Format**
```json
{
  "Decision": "Approved",
  "Amount": "450000", 
  "Justification": "Enhanced reasoning with hybrid sources...",
  "Relevant_Clauses": [
    {
      "clause_text": "Policy text from vector search",
      "document_id": "uploaded_doc", 
      "page_section": "chunk_3",
      "retrieval_source": "VDB"
    },
    {
      "clause_text": "Relationship from knowledge graph", 
      "document_id": "uploaded_doc",
      "page_section": "kg_entity",
      "retrieval_source": "KG"
    }
  ],
  "Processing_Info": {
    "processing_method": "gemini_multi_agent",
    "hybrid_mode": true,
    "vdb_chunks_found": 5,
    "kg_facts_found": 3,
    "model_used": "gemini-2.0-flash-lite"
  },
  "graph_data": {
    "nodes": 25,
    "edges": 18,
    "visualization_available": true
  }
}
```

## 🎯 Usage Examples

### **Enhanced Document Analysis**
```bash
# Upload document (via frontend or API)
curl -X POST "http://localhost:8000/upload_document" \
  -F "file=@insurance_policy.pdf"

# Enhanced query with hybrid search
curl -X POST "http://localhost:8000/process_query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Is knee surgery covered for a 45-year-old in Mumbai?"}'
```

### **Graph Exploration**
```bash
# Get complete graph structure
curl "http://localhost:8000/api/graph/structure"

# Get specific node types
curl "http://localhost:8000/api/graph/nodes/Policy"
```

## 🔧 Configuration

### **Environment Variables**
```bash
# Required
GEMINI_API_KEY=your_api_key

# Optional
GEMINI_MODEL=gemini-2.0-flash-lite
CHROMA_PERSIST_DIR=./data/chroma_db
NEO4J_URI=bolt://localhost:7687    # Optional
NEO4J_USERNAME=neo4j               # Optional  
NEO4J_PASSWORD=password            # Optional
```

### **Dependencies**
```bash
# Core requirements (hybrid_requirements.txt)
fastapi==0.104.1
google-generativeai==0.3.2
chromadb==0.4.18
sentence-transformers==2.2.2
langchain-community==0.0.10
# ... see hybrid_requirements.txt for complete list
```

## 🧪 Testing

### **Integration Test**
```bash
# Test complete system
python test_hybrid_integration.py
```

### **Manual Testing**
1. **Start both services**:
   ```bash
   # Terminal 1: Backend
   python hybrid_main.py
   
   # Terminal 2: Frontend  
   cd frontend && npm run dev
   ```

2. **Test workflow**:
   - Upload PDF via frontend
   - Ask questions in query interface
   - Explore graph visualization
   - Clear session and repeat

## 🎉 Benefits

### **For Users**
- **Better Accuracy**: Hybrid search finds more relevant information
- **Clearer Explanations**: Multi-agent reasoning provides better justifications
- **Richer Visualization**: See all document chunks and relationships
- **Faster Processing**: Gemini 2.0 Flash Lite has higher rate limits

### **For Developers**
- **No Breaking Changes**: Existing frontend works unchanged
- **Enhanced APIs**: More data and metadata in responses
- **Better Monitoring**: Comprehensive logging and status
- **Flexible Deployment**: No Docker required

## 🚨 Migration Guide

### **From Existing System**
1. **Install new dependencies**: `pip install -r hybrid_requirements.txt`
2. **Update environment**: Add `GEMINI_API_KEY` to `.env`
3. **Start enhanced backend**: `python hybrid_main.py` instead of `python main.py`
4. **Frontend unchanged**: No changes needed to React app

### **Fallback Mode**
- System automatically falls back if components unavailable
- Neo4j optional (uses in-memory storage)
- ChromaDB optional (degrades gracefully)
- Gemini optional (basic rule-based processing)

## 📈 Performance

### **Improvements**
- **Rate Limits**: 10x higher with Gemini 2.0 Flash Lite
- **Response Quality**: Hybrid search improves accuracy
- **Processing Speed**: Optimized multi-agent pipeline
- **Memory Usage**: Efficient ChromaDB and session management

### **Monitoring**
```bash
# Check system status
curl http://localhost:8000/status

# Monitor processing
tail -f logs/hybrid_rag_system.log
```

## 🔒 Security

### **Data Protection**
- **Session Isolation**: Each upload clears previous data
- **Local Processing**: Data stays on your machine
- **No Data Persistence**: Between sessions (optional)
- **API Security**: Input validation and error handling

## 🤝 Support

### **Common Issues**
1. **"GEMINI_API_KEY not found"**: Add API key to `.env` file
2. **"ChromaDB initialization failed"**: Check disk space and permissions
3. **"Neo4j connection failed"**: Optional - system uses fallback mode
4. **Rate limits**: Gemini 2.0 Flash Lite has much higher limits

### **Troubleshooting**
```bash
# Check system status
python test_hybrid_integration.py

# View detailed logs  
tail -f logs/hybrid_rag_system.log

# Reset everything
python -c "
import shutil
shutil.rmtree('data', ignore_errors=True)
shutil.rmtree('logs', ignore_errors=True)
"
```

---

## 🎯 Summary

The **Enhanced Hybrid RAG Integration** provides:

✅ **Seamless Integration**: Existing frontend works unchanged  
✅ **Enhanced Processing**: Vector + Knowledge Graph search  
✅ **Better Models**: Gemini 2.0 Flash Lite for higher rate limits  
✅ **Richer Visualization**: Interactive graph with more depth  
✅ **No Docker**: Simple local setup and deployment  
✅ **Session Management**: Clean data handling between documents  
✅ **Multi-Agent Reasoning**: Specialized AI agents for better decisions  

**Ready to use with your existing React frontend while providing significantly enhanced document processing capabilities!**