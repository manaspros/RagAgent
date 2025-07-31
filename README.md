# LLM Document Processing System

A sophisticated multi-agent system that leverages Knowledge Graphs and Google's Gemini API to process natural language queries against insurance policy documents. The system provides intelligent, transparent, and highly accurate decision-making for insurance claim processing.

## 🚀 Features

- **Multi-Agent Architecture**: Specialized agents for query parsing, knowledge graph analysis, policy reasoning, financial calculations, and decision synthesis
- **Knowledge Graph Integration**: Neo4j-based knowledge representation for superior reasoning and relationship understanding
- **Google Gemini API**: Advanced LLM capabilities for natural language processing and reasoning
- **RESTful API**: FastAPI-based backend with comprehensive documentation
- **Transparent Decision Making**: Clear justification and clause references for all decisions
- **Scalable Design**: Modular architecture supporting easy extension and customization

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│                 │    │                  │    │                 │
│   FastAPI App   │◄──►│  Multi-Agent     │◄──►│   Neo4j KG      │
│   (main.py)     │    │  Orchestrator    │    │   (kg_manager)  │
│                 │    │   (agents.py)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────▼────────┐             │
         │              │                 │             │
         └──────────────►│  Google Gemini  │◄────────────┘
                        │      API        │
                        └─────────────────┘
```

### Agent Workflow:
1. **QueryParsingAgent**: Extracts structured entities from natural language
2. **GraphQueryGenerationAgent**: Generates Cypher queries for Knowledge Graph
3. **KnowledgeGraphAnalysisAgent**: Executes queries and analyzes results
4. **PolicyReasoningAgent**: Performs logical inference and decision-making
5. **FinancialCalculationAgent**: Calculates payout amounts and financial details
6. **DecisionSynthesisAgent**: Synthesizes final decision with justification

## 📋 Prerequisites

- Python 3.9 or higher
- Neo4j Database (Community or Enterprise Edition)
- Google Gemini API key
- Git (for cloning the repository)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd RagAgent
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Neo4j Database

#### Option A: Docker (Recommended)
```bash
docker run \
    --name neo4j-insurance \
    -p7474:7474 -p7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/your_password_here \
    neo4j:latest
```

#### Option B: Local Installation
1. Download Neo4j from [https://neo4j.com/download/](https://neo4j.com/download/)
2. Install and start Neo4j
3. Access Neo4j Browser at `http://localhost:7474`
4. Set initial password

### 5. Configure Environment Variables

Copy the `.env` file and update with your credentials:

```bash
cp .env .env.local  # Optional: create local copy
```

Edit `.env` and set:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
NEO4J_PASSWORD=your_actual_neo4j_password
```

### 6. Get Google Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key to your `.env` file

## 🚀 Usage

### Starting the System

```bash
python main.py
```

The system will:
1. Initialize the Neo4j Knowledge Graph
2. Set up multi-agent orchestrator
3. Load sample policy documents
4. Start the FastAPI server on `http://localhost:8000`

### API Endpoints

#### 1. Process Query (Main Endpoint)
```bash
POST /process_query
Content-Type: application/json

{
    "query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
}
```

**Response:**
```json
{
    "Decision": "Approved",
    "Amount": 75000,
    "Justification": "Coverage approved for knee surgery. Patient meets age eligibility (46 years), procedure is covered under orthopedic benefits, and 90-day waiting period is satisfied for 3-month-old policy.",
    "Relevant_Clauses": [
        {
            "clause_text": "Orthopedic procedures including knee surgery are covered with a maximum limit of Rs. 75,000 per procedure",
            "document_id": "sample_policy.txt",
            "page_section": "Section 4.2 - Surgical Procedures"
        }
    ]
}
```

#### 2. System Status
```bash
GET /status
```

#### 3. Load Document
```bash
POST /load_document
Content-Type: application/json

{
    "document_path": "documents/new_policy.txt",
    "document_id": "policy_002"
}
```

#### 4. Knowledge Graph Schema
```bash
GET /kg_schema
```

### Interactive API Documentation

Access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📊 Example Queries

### Coverage Check
```json
{
    "query": "Can a 35-year-old female get heart surgery covered in Mumbai with a 6-month-old policy?"
}
```

### Claim Calculation
```json
{
    "query": "What is the payout for dental surgery for a 45-year-old in Delhi?"
}
```

### Eligibility Verification
```json
{
    "query": "Is cataract surgery covered for a 60-year-old person with 2-year-old insurance?"
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `NEO4J_URI` | Neo4j database URI | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | Required |
| `HOST` | FastAPI server host | `127.0.0.1` |
| `PORT` | FastAPI server port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Customizing Agents

Each agent can be customized by modifying the respective class in `agents.py`:

```python
class CustomQueryParsingAgent(BaseAgent):
    def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        # Custom implementation
        pass
```

### Adding New Document Types

1. Place documents in the `documents/` directory
2. Use the `/load_document` endpoint to process them
3. The system will automatically extract entities and relationships

## 🧪 Testing

### Running Tests
```bash
pytest tests/
```

### Manual Testing
Use the provided example queries or test with curl:

```bash
curl -X POST "http://localhost:8000/process_query" \
     -H "Content-Type: application/json" \
     -d '{"query": "knee surgery for 30-year-old in Mumbai with 4-month policy"}'
```

## 📈 Monitoring and Logging

The system provides comprehensive logging:

- **Application Logs**: General system operations
- **Agent Logs**: Individual agent processing steps
- **Neo4j Logs**: Database operations and queries
- **API Logs**: Request/response logging

Logs are output to console by default. Configure logging in `main.py` for file output.

## 🔒 Security Considerations

- **API Keys**: Store securely in environment variables
- **Database**: Use strong passwords and secure connections
- **CORS**: Configure appropriately for production
- **Input Validation**: All inputs are validated using Pydantic models

## 🚀 Production Deployment

### Using Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Using Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - neo4j
    environment:
      - NEO4J_URI=bolt://neo4j:7687
  
  neo4j:
    image: neo4j:latest
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

#### Neo4j Connection Failed
- Ensure Neo4j is running
- Check connection credentials
- Verify port 7687 is accessible

#### Gemini API Errors
- Verify API key is correct
- Check API quotas and limits
- Ensure internet connectivity

#### Agent Processing Errors
- Check input query format
- Verify Knowledge Graph has data
- Review agent logs for specific errors

### Getting Help

- Create an issue on GitHub
- Check the logs for detailed error messages
- Verify all environment variables are set correctly

## 📚 Additional Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)