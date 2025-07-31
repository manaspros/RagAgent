"""
Test Neo4j Connection
Run this after setting up Neo4j to verify connectivity
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

def test_neo4j_connection():
    """Test Neo4j connection with current configuration"""
    
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password123")
    
    print(f"Testing Neo4j connection...")
    print(f"URI: {uri}")
    print(f"Username: {username}")
    print(f"Password: {'*' * len(password)}")
    print("-" * 50)
    
    try:
        # Create driver
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful!' as message, datetime() as timestamp")
            record = result.single()
            
            print("✅ SUCCESS!")
            print(f"Message: {record['message']}")
            print(f"Timestamp: {record['timestamp']}")
            
            # Test database info
            result = session.run("CALL db.info()")
            info = result.single()
            if info:
                print(f"Database: {info}")
            
            # Test creating a simple node
            session.run("CREATE (test:TestNode {name: 'Connection Test', created: datetime()})")
            result = session.run("MATCH (test:TestNode) RETURN count(test) as node_count")
            count = result.single()["node_count"]
            print(f"Test nodes created: {count}")
            
            # Clean up test node
            session.run("MATCH (test:TestNode) DELETE test")
            print("Test cleanup completed")
            
        driver.close()
        print("\n🎉 Neo4j is ready for your LLM Document Processing System!")
        return True
        
    except Exception as e:
        print(f"❌ CONNECTION FAILED!")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Neo4j Desktop is running")
        print("2. Check if the database is started (green play button)")
        print("3. Verify password matches your Neo4j database password")
        print("4. Ensure port 7687 is not blocked by firewall")
        return False

if __name__ == "__main__":
    test_neo4j_connection()