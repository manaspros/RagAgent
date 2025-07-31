"""
Rate Limiting Configuration for LLM Document Processing System
Centralized configuration for API rate limiting across all components
"""

import os

class RateLimitConfig:
    """Configuration for rate limiting across the system"""
    
    # Base delays for different components (in seconds)
    QUERY_PARSING_AGENT_DELAY = float(os.getenv("QUERY_PARSING_DELAY", "2.0"))
    GRAPH_QUERY_GEN_DELAY = float(os.getenv("GRAPH_QUERY_GEN_DELAY", "2.5"))
    KG_ANALYSIS_DELAY = float(os.getenv("KG_ANALYSIS_DELAY", "3.0"))
    POLICY_REASONING_DELAY = float(os.getenv("POLICY_REASONING_DELAY", "3.5"))
    FINANCIAL_CALC_DELAY = float(os.getenv("FINANCIAL_CALC_DELAY", "4.0"))
    DECISION_SYNTHESIS_DELAY = float(os.getenv("DECISION_SYNTHESIS_DELAY", "4.5"))
    
    # Document processor delay (longer since it processes large documents)
    DOCUMENT_PROCESSOR_DELAY = float(os.getenv("DOCUMENT_PROCESSOR_DELAY", "8.0"))
    
    # Inter-agent delay (delay between different agents in orchestrator)
    INTER_AGENT_DELAY = float(os.getenv("INTER_AGENT_DELAY", "1.5"))
    
    # Retry configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    
    # Backoff multiplier for exponential backoff
    BACKOFF_MULTIPLIER = float(os.getenv("BACKOFF_MULTIPLIER", "2.0"))
    
    # Jitter range (random delay addition)
    MIN_JITTER = float(os.getenv("MIN_JITTER", "0.5"))
    MAX_JITTER = float(os.getenv("MAX_JITTER", "2.0"))
    
    # Document processing jitter (higher for document processing)
    DOC_MIN_JITTER = float(os.getenv("DOC_MIN_JITTER", "2.0"))
    DOC_MAX_JITTER = float(os.getenv("DOC_MAX_JITTER", "5.0"))
    
    @classmethod
    def get_agent_delay(cls, agent_name: str) -> float:
        """Get the appropriate delay for a specific agent"""
        delay_map = {
            "QueryParsingAgent": cls.QUERY_PARSING_AGENT_DELAY,
            "GraphQueryGenerationAgent": cls.GRAPH_QUERY_GEN_DELAY,
            "KnowledgeGraphAnalysisAgent": cls.KG_ANALYSIS_DELAY,
            "PolicyReasoningAgent": cls.POLICY_REASONING_DELAY,
            "FinancialCalculationAgent": cls.FINANCIAL_CALC_DELAY,
            "DecisionSynthesisAgent": cls.DECISION_SYNTHESIS_DELAY,
        }
        return delay_map.get(agent_name, 3.0)  # Default 3 seconds
    
    @classmethod
    def get_rate_limit_info(cls) -> dict:
        """Get current rate limiting configuration"""
        return {
            "agent_delays": {
                "query_parsing": cls.QUERY_PARSING_AGENT_DELAY,
                "graph_query_gen": cls.GRAPH_QUERY_GEN_DELAY,
                "kg_analysis": cls.KG_ANALYSIS_DELAY,
                "policy_reasoning": cls.POLICY_REASONING_DELAY,
                "financial_calc": cls.FINANCIAL_CALC_DELAY,
                "decision_synthesis": cls.DECISION_SYNTHESIS_DELAY,
                "document_processor": cls.DOCUMENT_PROCESSOR_DELAY,
            },
            "orchestrator": {
                "inter_agent_delay": cls.INTER_AGENT_DELAY,
            },
            "retry_config": {
                "max_retries": cls.MAX_RETRIES,
                "backoff_multiplier": cls.BACKOFF_MULTIPLIER,
            },
            "jitter": {
                "min_jitter": cls.MIN_JITTER,
                "max_jitter": cls.MAX_JITTER,
                "doc_min_jitter": cls.DOC_MIN_JITTER,
                "doc_max_jitter": cls.DOC_MAX_JITTER,
            }
        }