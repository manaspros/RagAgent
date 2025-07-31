#!/usr/bin/env python3
"""
Hybrid RAG System Setup - No Docker Required
Sets up the enhanced system with Gemini models and ChromaDB
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any

def print_banner():
    """Print setup banner"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          🚀 Enhanced Hybrid RAG System Setup                ║
║                                                              ║
║  Features:                                                   ║
║  • Google Gemini 2.0 Flash Lite (High Rate Limits)         ║
║  • ChromaDB Vector Database (Local)                         ║
║  • Neo4j Knowledge Graph (Optional)                         ║
║  • Existing Frontend Integration                            ║
║  • No Docker Required                                       ║
╚══════════════════════════════════════════════════════════════╝
""")

def check_python_version() -> bool:
    """Check Python version"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 9):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} not supported")
        print("   Please install Python 3.9 or higher")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directory structure...")
    
    directories = [
        "data",
        "data/chroma_db",
        "documents/uploads",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}/")
    
    print("✅ Directory structure created")

def install_dependencies() -> bool:
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        if not Path("hybrid_requirements.txt").exists():
            print("❌ hybrid_requirements.txt not found")
            return False
        
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "hybrid_requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_environment():
    """Setup environment configuration"""
    print("⚙️ Setting up environment...")
    
    env_content = """# Enhanced Hybrid RAG System Configuration

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-lite

# Neo4j Configuration (Optional - system works in fallback mode)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# ChromaDB Configuration
CHROMA_PERSIST_DIR=./data/chroma_db

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# System Configuration
LOG_LEVEL=INFO
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(env_content)
        print("✅ Created .env configuration file")
        print("   ⚠️  Please edit .env with your actual API keys!")
    else:
        print("   .env file already exists")

def test_system_components():
    """Test system components"""
    print("🔍 Testing system components...")
    
    try:
        # Test ChromaDB
        import chromadb
        print("   ✅ ChromaDB available")
    except ImportError:
        print("   ❌ ChromaDB not available")
    
    try:
        # Test sentence transformers
        from sentence_transformers import SentenceTransformer
        print("   ✅ Sentence Transformers available")
    except ImportError:
        print("   ❌ Sentence Transformers not available")
    
    try:
        # Test Google Gemini
        import google.generativeai as genai
        print("   ✅ Google Gemini API available")
    except ImportError:
        print("   ❌ Google Gemini API not available")
    
    try:
        # Test Neo4j (optional)
        from neo4j import GraphDatabase
        print("   ✅ Neo4j driver available (optional)")
    except ImportError:
        print("   ⚠️  Neo4j driver not available (will use fallback mode)")
    
    try:
        # Test FastAPI
        import fastapi
        print("   ✅ FastAPI available")
    except ImportError:
        print("   ❌ FastAPI not available")

def create_startup_script():
    """Create startup script"""
    print("📝 Creating startup script...")
    
    startup_content = '''#!/usr/bin/env python3
"""
Hybrid RAG System Startup Script
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 Starting Enhanced Hybrid RAG System...")
    
    # Check if .env exists
    if not Path(".env").exists():
        print("❌ .env file not found. Please run setup first.")
        return
    
    # Check if hybrid_main.py exists
    if not Path("hybrid_main.py").exists():
        print("❌ hybrid_main.py not found.")
        return
    
    try:
        # Start the system
        print("🌐 Starting server at http://localhost:8000")
        print("📚 API docs will be at http://localhost:8000/docs")
        print("🔍 Graph visualization via API endpoints")
        print("\\nPress Ctrl+C to stop the server\\n")
        
        subprocess.run([sys.executable, "hybrid_main.py"])
        
    except KeyboardInterrupt:
        print("\\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open("start_hybrid_system.py", "w", encoding="utf-8") as f:
        f.write(startup_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod("start_hybrid_system.py", 0o755)
    
    print("✅ Created start_hybrid_system.py")

def show_next_steps():
    """Show next steps"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("""
📝 NEXT STEPS:

1. 🔑 Configure API Key:
   Edit .env file and add your Gemini API key:
   GEMINI_API_KEY=your_actual_api_key_here

2. 🚀 Start the System:
   python start_hybrid_system.py
   
   Or manually:
   python hybrid_main.py

3. 🌐 Access the System:
   • Frontend: http://localhost:3000 (existing React app)
   • API: http://localhost:8000
   • Docs: http://localhost:8000/docs

4. 📄 Upload Documents:
   Use the existing frontend to upload PDF documents
   They will be processed with the hybrid RAG system

5. 🔍 Ask Questions:
   Use the existing query interface
   Enhanced with vector + knowledge graph search

🎯 INTEGRATION FEATURES:
• Existing frontend works unchanged
• Enhanced with ChromaDB semantic search  
• Knowledge graph visualization
• Gemini 2.0 Flash Lite (higher rate limits)
• Session-based processing
• No Docker required

💡 TROUBLESHOOTING:
• Check .env configuration
• Ensure Gemini API key is valid
• Check logs/ directory for detailed logs
• Neo4j is optional (fallback mode available)
""")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Dependency installation failed")
        print("   You may need to install manually:")
        print("   pip install -r hybrid_requirements.txt")
    
    # Setup environment
    setup_environment()
    
    # Test components
    test_system_components()
    
    # Create startup script
    create_startup_script()
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Setup interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Setup error: {e}")
        sys.exit(1)