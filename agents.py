"""
Multi-Agent System for LLM Document Processing
Implements specialized agents for query processing, knowledge graph analysis,
and decision synthesis using Google's Gemini API
"""

import os
import json
import logging
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re
from abc import ABC, abstractmethod

import google.generativeai as genai
from kg_manager import EnhancedKnowledgeGraphManager
from rate_limiting_config import RateLimitConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


@dataclass
class AgentResponse:
    """Standard response format for all agents"""
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None
    agent_name: Optional[str] = None


class BaseAgent(ABC):
    """Abstract base class for all agents with rate limiting"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-lite", base_delay: float = 2.0):
        self.model = genai.GenerativeModel(model_name)
        self.agent_name = self.__class__.__name__
        self.base_delay = base_delay  # Base delay between API calls
        self.last_api_call = 0  # Track last API call time
        
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Process input and return structured response"""
        pass
    
    def _generate_response(self, prompt: str, max_retries: int = 3) -> str:
        """Generate response using Gemini API with rate limiting and retries"""
        for attempt in range(max_retries):
            try:
                # Rate limiting: ensure minimum delay between API calls
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call
                
                if time_since_last_call < self.base_delay:
                    sleep_time = self.base_delay - time_since_last_call
                    # Add some jitter to avoid thundering herd
                    sleep_time += random.uniform(RateLimitConfig.MIN_JITTER, RateLimitConfig.MAX_JITTER)
                    logger.info(f"{self.agent_name}: Waiting {sleep_time:.2f}s to avoid rate limit")
                    time.sleep(sleep_time)
                
                # Make API call
                logger.info(f"{self.agent_name}: Making API call (attempt {attempt + 1}/{max_retries})")
                response = self.model.generate_content(prompt)
                self.last_api_call = time.time()
                
                return response.text
                
            except Exception as e:
                error_message = str(e)
                logger.warning(f"{self.agent_name} attempt {attempt + 1} failed: {error_message}")
                
                # Check if it's a rate limit error
                if "rate limit" in error_message.lower() or "quota" in error_message.lower() or "429" in error_message:
                    if attempt < max_retries - 1:
                        # Exponential backoff for rate limit errors
                        backoff_time = (RateLimitConfig.BACKOFF_MULTIPLIER ** attempt) * self.base_delay + random.uniform(RateLimitConfig.MIN_JITTER, RateLimitConfig.MAX_JITTER * 2)
                        logger.info(f"{self.agent_name}: Rate limit hit, backing off for {backoff_time:.2f}s")
                        time.sleep(backoff_time)
                        continue
                    else:
                        logger.error(f"{self.agent_name}: Rate limit exceeded after {max_retries} attempts")
                        raise Exception(f"Rate limit exceeded: {error_message}")
                else:
                    # For non-rate-limit errors, fail immediately
                    logger.error(f"Error in {self.agent_name}: {e}")
                    raise
        
        raise Exception(f"Max retries ({max_retries}) exceeded for {self.agent_name}")


class QueryParsingAgent(BaseAgent):
    """
    Agent responsible for parsing natural language queries and extracting
    structured entities and intent
    """
    
    def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Parse natural language query into structured format
        
        Args:
            input_data: {"query": str}
            
        Returns:
            AgentResponse with parsed entities and intent
        """
        try:
            query = input_data.get("query", "")
            if not query:
                return AgentResponse(
                    success=False,
                    data={},
                    error_message="No query provided",
                    agent_name=self.agent_name
                )
            
            parsing_prompt = f"""
            You are a query parsing agent for an insurance policy processing system.
            
            Parse the following natural language query and extract structured information:
            Query: "{query}"
            
            Extract the following entities and return as JSON:
            {{
                "intent": "check_coverage" | "calculate_payout" | "verify_eligibility" | "general_inquiry",
                "person": {{
                    "age": number or null,
                    "gender": "male" | "female" | null,
                    "location": string or null
                }},
                "medical_info": {{
                    "procedure": string or null,
                    "condition": string or null,
                    "urgency": "emergency" | "planned" | null
                }},
                "policy_info": {{
                    "policy_age_months": number or null,
                    "policy_type": string or null
                }},
                "financial_info": {{
                    "expected_cost": number or null,
                    "currency": "INR" | "USD" | null
                }},
                "extracted_keywords": [list of important keywords]
            }}
            
            Be precise and only extract information that is explicitly mentioned or can be reasonably inferred.
            If information is not available, use null values.
            """
            
            response_text = self._generate_response(parsing_prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
            else:
                # Fallback parsing
                parsed_data = self._fallback_parse(query)
            
            return AgentResponse(
                success=True,
                data=parsed_data,
                agent_name=self.agent_name
            )
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            return AgentResponse(
                success=False,
                data={},
                error_message=str(e),
                agent_name=self.agent_name
            )
    
    def _fallback_parse(self, query: str) -> Dict[str, Any]:
        """Fallback parsing using regex patterns"""
        parsed = {
            "intent": "general_inquiry",
            "person": {"age": None, "gender": None, "location": None},
            "medical_info": {"procedure": None, "condition": None, "urgency": None},
            "policy_info": {"policy_age_months": None, "policy_type": None},
            "financial_info": {"expected_cost": None, "currency": "INR"},
            "extracted_keywords": []
        }
        
        # Extract age
        age_match = re.search(r'(\d+)[-\s]*year[-\s]*old', query, re.IGNORECASE)
        if age_match:
            parsed["person"]["age"] = int(age_match.group(1))
        
        # Extract gender
        if re.search(r'\bmale\b', query, re.IGNORECASE):
            parsed["person"]["gender"] = "male"
        elif re.search(r'\bfemale\b', query, re.IGNORECASE):
            parsed["person"]["gender"] = "female"
        
        # Extract procedures
        procedures = ["surgery", "treatment", "consultation", "knee surgery", "heart surgery"]
        for proc in procedures:
            if proc.lower() in query.lower():
                parsed["medical_info"]["procedure"] = proc.title()
                break
        
        # Extract locations
        locations = ["mumbai", "delhi", "pune", "bangalore", "chennai"]
        for loc in locations:
            if loc.lower() in query.lower():
                parsed["person"]["location"] = loc.title()
                break
        
        # Extract policy age
        policy_age_match = re.search(r'(\d+)[-\s]*month[-\s]*old\s+(?:insurance|policy)', query, re.IGNORECASE)
        if policy_age_match:
            parsed["policy_info"]["policy_age_months"] = int(policy_age_match.group(1))
        
        # Determine intent
        if any(word in query.lower() for word in ["cover", "coverage", "eligible"]):
            parsed["intent"] = "check_coverage"
        elif any(word in query.lower() for word in ["cost", "amount", "payout", "claim"]):
            parsed["intent"] = "calculate_payout"
        
        return parsed


class GraphQueryGenerationAgent(BaseAgent):
    """
    Agent responsible for generating Cypher queries based on parsed query data
    """
    
    def __init__(self, kg_manager: EnhancedKnowledgeGraphManager, model_name: str = "gemini-2.0-flash-lite", base_delay: float = 2.5):
        super().__init__(model_name, base_delay)
        self.kg_manager = kg_manager
        self.schema_info = None
    
    def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Generate Cypher query based on parsed query data
        
        Args:
            input_data: {"parsed_query": Dict, "intent": str}
            
        Returns:
            AgentResponse with generated Cypher query
        """
        try:
            parsed_query = input_data.get("parsed_query", {})
            intent = parsed_query.get("intent", "general_inquiry")
            
            # Get KG schema if not cached
            if not self.schema_info:
                self.schema_info = self.kg_manager.get_kg_schema()
            
            cypher_prompt = f"""
            You are a Cypher query generation agent for a Neo4j knowledge graph containing insurance policy information.
            
            Knowledge Graph Schema:
            - Node Labels: {self.schema_info.get('node_labels', [])}
            - Relationship Types: {self.schema_info.get('relationship_types', [])}
            - Property Keys: {self.schema_info.get('property_keys', [])}
            
            Based on this parsed query data:
            {json.dumps(parsed_query, indent=2)}
            
            Intent: {intent}
            
            Generate a Cypher query to retrieve relevant information. Consider:
            
            1. For "check_coverage" intent: Find policies that cover the specified procedure
            2. For "calculate_payout" intent: Find coverage limits and conditions
            3. For "verify_eligibility" intent: Check age and other eligibility conditions
            
            Common patterns:
            - Match policies: MATCH (p:Policy)
            - Match procedures: MATCH (pr:Procedure)
            - Match coverage: MATCH (p:Policy)-[:COVERS_PROCEDURE]->(pr:Procedure)
            - Match conditions: MATCH (p:Policy)-[:HAS_ELIGIBILITY_CONDITION]->(c:Condition)
            
            Return only the Cypher query, no explanation.
            """
            
            cypher_query = self._generate_response(cypher_prompt).strip()
            
            # Clean up the query
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            
            return AgentResponse(
                success=True,
                data={"cypher_query": cypher_query},
                agent_name=self.agent_name
            )
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            return AgentResponse(
                success=False,
                data={},
                error_message=str(e),
                agent_name=self.agent_name
            )


class KnowledgeGraphAnalysisAgent(BaseAgent):
    """
    Agent responsible for executing Cypher queries and analyzing KG results
    """
    
    def __init__(self, kg_manager: EnhancedKnowledgeGraphManager, model_name: str = "gemini-2.0-flash-lite", base_delay: float = 3.0):
        super().__init__(model_name, base_delay)
        self.kg_manager = kg_manager
    
    def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Execute Cypher query and analyze results
        
        Args:
            input_data: {"cypher_query": str, "parsed_query": Dict}
            
        Returns:
            AgentResponse with KG findings and analysis
        """
        try:
            cypher_query = input_data.get("cypher_query", "")
            parsed_query = input_data.get("parsed_query", {})
            
            if not cypher_query:
                return AgentResponse(
                    success=False,
                    data={},
                    error_message="No Cypher query provided",
                    agent_name=self.agent_name
                )
            
            # Execute Cypher query
            kg_results = self.kg_manager.query_kg(cypher_query)
            
            # Analyze results using chain-of-thought
            analysis_prompt = f"""
            You are a knowledge graph analysis agent. Analyze the following graph query results
            in the context of the original query.
            
            Original Query Context:
            {json.dumps(parsed_query, indent=2)}
            
            Cypher Query Executed:
            {cypher_query}
            
            Graph Results:
            {json.dumps(kg_results, indent=2)}
            
            Please provide a chain-of-thought analysis following this structure:
            
            1. **Graph Data Summary**: What entities and relationships were found?
            2. **Relevance Analysis**: How do these results relate to the original query?
            3. **Key Findings**: What are the most important facts for decision making?
            4. **Identified Entities**: List specific entities with their properties
            5. **Missing Information**: What information might be needed but wasn't found?
            
            Return your analysis as JSON:
            {{
                "kg_findings": "natural language summary of graph results",
                "identified_entities": [
                    {{
                        "type": "Policy|Procedure|Condition|Location",
                        "name": "entity name",
                        "properties": {{key: value pairs}}
                    }}
                ],
                "key_relationships": [
                    {{
                        "source": "entity1",
                        "relationship": "relationship_type", 
                        "target": "entity2",
                        "properties": {{}}
                    }}
                ],
                "relevance_score": 0.0-1.0,
                "missing_info": ["list of missing information"],
                "confidence": 0.0-1.0
            }}
            """
            
            analysis_response = self._generate_response(analysis_prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', analysis_response, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                # Fallback analysis
                analysis_data = self._fallback_analysis(kg_results, parsed_query)
            
            return AgentResponse(
                success=True,
                data=analysis_data,
                agent_name=self.agent_name
            )
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            return AgentResponse(
                success=False,
                data={},
                error_message=str(e),
                agent_name=self.agent_name
            )
    
    def _fallback_analysis(self, kg_results: List[Dict], parsed_query: Dict) -> Dict:
        """Fallback analysis when LLM parsing fails"""
        return {
            "kg_findings": f"Found {len(kg_results)} results in knowledge graph",
            "identified_entities": [],
            "key_relationships": [],
            "relevance_score": 0.7,
            "missing_info": [],
            "confidence": 0.6
        }


class PolicyReasoningAgent(BaseAgent):
    """
    Agent responsible for policy reasoning and decision logic
    """
    
    def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Perform policy reasoning based on KG findings
        
        Args:
            input_data: {"parsed_query": Dict, "kg_analysis": Dict}
            
        Returns:
            AgentResponse with reasoning and decision logic
        """
        try:
            parsed_query = input_data.get("parsed_query", {})
            kg_analysis = input_data.get("kg_analysis", {})
            
            reasoning_prompt = f"""
            You are a policy reasoning agent for insurance claim processing.
            
            Query Context:
            {json.dumps(parsed_query, indent=2)}
            
            Knowledge Graph Analysis:
            {json.dumps(kg_analysis, indent=2)}
            
            Perform step-by-step reasoning to determine:
            1. Eligibility based on age, location, policy duration
            2. Coverage applicability for the requested procedure
            3. Any waiting periods or conditions that apply
            4. Potential conflicts or gaps in coverage
            
            Use this reasoning framework:
            
            Step 1: Eligibility Check
            - Age requirements: Is the person within eligible age range?
            - Policy duration: Has the policy been active long enough?
            - Geographic coverage: Is the location covered?
            
            Step 2: Procedure Coverage Analysis  
            - Is the procedure covered under the policy?
            - What are the coverage limits?
            - Are there any exclusions?
            
            Step 3: Waiting Period Assessment
            - Has the waiting period been satisfied?
            - Are there different waiting periods for different procedures?
            
            Step 4: Final Decision Logic
            - Combine all factors to reach a decision
            - Identify any conflicts or ambiguities
            
            Return as JSON:
            {{
                "decision_logic_steps": [
                    "Step 1: ...",
                    "Step 2: ...",
                    "Step 3: ...",
                    "Step 4: ..."
                ],
                "applicable_rules": [
                    "rule text or reference"
                ],
                "eligibility_status": "eligible" | "not_eligible" | "conditional",
                "coverage_status": "covered" | "not_covered" | "partial",
                "waiting_period_status": "satisfied" | "not_satisfied" | "n/a",
                "gaps_or_conflicts": [
                    "description of any gaps or conflicts"
                ],
                "confidence_score": 0.0-1.0
            }}
            """
            
            reasoning_response = self._generate_response(reasoning_prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', reasoning_response, re.DOTALL)
            if json_match:
                reasoning_data = json.loads(json_match.group())
            else:
                # Fallback reasoning
                reasoning_data = self._fallback_reasoning(parsed_query, kg_analysis)
            
            return AgentResponse(
                success=True,
                data=reasoning_data,
                agent_name=self.agent_name
            )
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            return AgentResponse(
                success=False,
                data={},
                error_message=str(e),
                agent_name=self.agent_name
            )
    
    def _fallback_reasoning(self, parsed_query: Dict, kg_analysis: Dict) -> Dict:
        """Fallback reasoning logic"""
        return {
            "decision_logic_steps": [
                "Step 1: Checking basic eligibility criteria",
                "Step 2: Analyzing procedure coverage",
                "Step 3: Evaluating waiting periods",
                "Step 4: Making final determination"
            ],
            "applicable_rules": ["Standard policy terms apply"],
            "eligibility_status": "conditional",
            "coverage_status": "partial",
            "waiting_period_status": "satisfied",
            "gaps_or_conflicts": [],
            "confidence_score": 0.6
        }


class FinancialCalculationAgent(BaseAgent):
    """
    Agent responsible for financial calculations and payout determination
    """
    
    def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Calculate payout amounts based on policy reasoning
        
        Args:
            input_data: {"parsed_query": Dict, "policy_reasoning": Dict, "kg_analysis": Dict}
            
        Returns:
            AgentResponse with financial calculations
        """
        try:
            parsed_query = input_data.get("parsed_query", {})
            policy_reasoning = input_data.get("policy_reasoning", {})
            kg_analysis = input_data.get("kg_analysis", {})
            
            calculation_prompt = f"""
            You are a financial calculation agent for insurance claim processing.
            
            Query Context:
            {json.dumps(parsed_query, indent=2)}
            
            Policy Reasoning Results:
            {json.dumps(policy_reasoning, indent=2)}
            
            Knowledge Graph Analysis:
            {json.dumps(kg_analysis, indent=2)}
            
            Calculate the financial aspects:
            
            1. If coverage is approved, determine payout amount based on:
               - Coverage limits from policy
               - Procedure-specific limits
               - Deductibles or co-payments
               - Geographic adjustments
            
            2. Consider factors like:
               - Sum assured limits
               - Sub-limits for specific procedures
               - Room rent limits
               - Pre and post hospitalization coverage
            
            3. Apply any applicable deductions:
               - Deductibles
               - Co-insurance percentages
               - Waiting period penalties
            
            Return as JSON:
            {{
                "calculated_amount": number or "N/A",
                "currency": "INR",
                "calculation_details": "step by step calculation explanation",
                "coverage_limits": {{
                    "sum_assured": number,
                    "procedure_limit": number,
                    "room_rent_limit": number
                }},
                "deductions": {{
                    "deductible": number,
                    "co_insurance": number,
                    "total_deductions": number
                }},
                "final_payout": number or "N/A",
                "calculation_confidence": 0.0-1.0
            }}
            
            If coverage is not approved, return "N/A" for amounts.
            """
            
            calculation_response = self._generate_response(calculation_prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', calculation_response, re.DOTALL)
            if json_match:
                calculation_data = json.loads(json_match.group())
            else:
                # Fallback calculation
                calculation_data = self._fallback_calculation(policy_reasoning)
            
            return AgentResponse(
                success=True,
                data=calculation_data,
                agent_name=self.agent_name
            )
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            return AgentResponse(
                success=False,
                data={},
                error_message=str(e),
                agent_name=self.agent_name
            )
    
    def _fallback_calculation(self, policy_reasoning: Dict) -> Dict:
        """Fallback calculation logic"""
        if policy_reasoning.get("coverage_status") == "covered":
            return {
                "calculated_amount": 75000,
                "currency": "INR",
                "calculation_details": "Standard coverage amount for approved procedures",
                "coverage_limits": {"sum_assured": 500000, "procedure_limit": 75000, "room_rent_limit": 5000},
                "deductions": {"deductible": 0, "co_insurance": 0, "total_deductions": 0},
                "final_payout": 75000,
                "calculation_confidence": 0.7
            }
        else:
            return {
                "calculated_amount": "N/A",
                "currency": "INR", 
                "calculation_details": "Coverage not approved",
                "coverage_limits": {},
                "deductions": {},
                "final_payout": "N/A",
                "calculation_confidence": 1.0
            }


class DecisionSynthesisAgent(BaseAgent):
    """
    Agent responsible for synthesizing all outputs into final decision
    """
    
    def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Synthesize all agent outputs into final decision
        
        Args:
            input_data: Dict containing all previous agent outputs
            
        Returns:
            AgentResponse with final structured decision
        """
        try:
            synthesis_prompt = f"""
            You are a decision synthesis agent responsible for creating the final response
            for an insurance claim processing system.
            
            All Agent Outputs:
            {json.dumps(input_data, indent=2)}
            
            Synthesize all information into a final decision following this format:
            
            {{
                "Decision": "Approved" | "Rejected" | "Requires Further Review",
                "Amount": number or "N/A",
                "Justification": "clear, comprehensive explanation of the decision",
                "Relevant_Clauses": [
                    {{
                        "clause_text": "text of relevant policy clause",
                        "document_id": "source document identifier",
                        "page_section": "section reference"
                    }}
                ]
            }}
            
            Guidelines:
            1. Decision should be "Approved" only if all conditions are met
            2. Decision should be "Rejected" if key requirements are not satisfied
            3. Decision should be "Requires Further Review" if there are ambiguities
            4. Amount should match the financial calculation or be "N/A" if rejected
            5. Justification should be clear, referencing specific policy terms
            6. Include relevant clauses that support the decision
            
            Make the justification persuasive and easy to understand.
            """
            
            synthesis_response = self._generate_response(synthesis_prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', synthesis_response, re.DOTALL)
            if json_match:
                synthesis_data = json.loads(json_match.group())
            else:
                # Fallback synthesis
                synthesis_data = self._fallback_synthesis(input_data)
            
            return AgentResponse(
                success=True,
                data=synthesis_data,
                agent_name=self.agent_name
            )
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            return AgentResponse(
                success=False,
                data={},
                error_message=str(e),
                agent_name=self.agent_name
            )
    
    def _fallback_synthesis(self, input_data: Dict) -> Dict:
        """Fallback synthesis logic"""
        policy_reasoning = input_data.get("policy_reasoning", {})
        financial_calc = input_data.get("financial_calculation", {})
        
        if policy_reasoning.get("coverage_status") == "covered":
            decision = "Approved"
            amount = financial_calc.get("final_payout", "N/A")
            justification = "Coverage approved based on policy terms and eligibility criteria."
        else:
            decision = "Rejected"
            amount = "N/A"
            justification = "Coverage not approved due to policy restrictions or eligibility issues."
        
        return {
            "Decision": decision,
            "Amount": amount,
            "Justification": justification,
            "Relevant_Clauses": [
                {
                    "clause_text": "Standard policy terms and conditions apply",
                    "document_id": "sample_policy.txt",
                    "page_section": "Section 1"
                }
            ]
        }


class MultiAgentOrchestrator:
    """
    Orchestrates the multi-agent system workflow with rate limiting
    """
    
    def __init__(self, kg_manager: EnhancedKnowledgeGraphManager, inter_agent_delay: float = None):
        self.kg_manager = kg_manager
        self.inter_agent_delay = inter_agent_delay or RateLimitConfig.INTER_AGENT_DELAY
        
        # Create agents with different base delays to stagger API calls
        self.agents = {
            "query_parser": QueryParsingAgent(base_delay=RateLimitConfig.QUERY_PARSING_AGENT_DELAY),
            "graph_query_gen": GraphQueryGenerationAgent(kg_manager, base_delay=RateLimitConfig.GRAPH_QUERY_GEN_DELAY),
            "kg_analysis": KnowledgeGraphAnalysisAgent(kg_manager, base_delay=RateLimitConfig.KG_ANALYSIS_DELAY),
            "policy_reasoning": PolicyReasoningAgent(base_delay=RateLimitConfig.POLICY_REASONING_DELAY),
            "financial_calc": FinancialCalculationAgent(base_delay=RateLimitConfig.FINANCIAL_CALC_DELAY),
            "decision_synthesis": DecisionSynthesisAgent(base_delay=RateLimitConfig.DECISION_SYNTHESIS_DELAY)
        }
    
    def process_multi_agent_query(self, query: str) -> Dict[str, Any]:
        """
        Process query through multi-agent workflow
        
        Args:
            query: Natural language query
            
        Returns:
            Final structured response
        """
        try:
            logger.info(f"Processing query: {query}")
            workflow_data = {"original_query": query}
            
            # Step 1: Parse Query
            logger.info("Step 1: Parsing query...")
            parse_response = self.agents["query_parser"].process({"query": query})
            if not parse_response.success:
                return {"error": f"Query parsing failed: {parse_response.error_message}"}
            
            workflow_data["parsed_query"] = parse_response.data
            logger.info(f"Parsed query: {parse_response.data.get('intent', 'unknown')}")
            
            # Inter-agent delay
            time.sleep(self.inter_agent_delay)
            
            # Step 2: Generate Cypher Query
            logger.info("Step 2: Generating Cypher query...")
            cypher_response = self.agents["graph_query_gen"].process({
                "parsed_query": parse_response.data,
                "intent": parse_response.data.get("intent")
            })
            if not cypher_response.success:
                return {"error": f"Cypher generation failed: {cypher_response.error_message}"}
            
            workflow_data["cypher_query"] = cypher_response.data.get("cypher_query")
            logger.info(f"Generated Cypher: {workflow_data['cypher_query']}")
            
            # Inter-agent delay
            time.sleep(self.inter_agent_delay)
            
            # Step 3: Analyze Knowledge Graph
            logger.info("Step 3: Analyzing knowledge graph...")
            kg_response = self.agents["kg_analysis"].process({
                "cypher_query": workflow_data["cypher_query"],
                "parsed_query": parse_response.data
            })
            if not kg_response.success:
                return {"error": f"KG analysis failed: {kg_response.error_message}"}
            
            workflow_data["kg_analysis"] = kg_response.data
            logger.info(f"KG Analysis complete: {kg_response.data.get('relevance_score', 0)}")
            
            # Inter-agent delay
            time.sleep(self.inter_agent_delay)
            
            # Step 4: Policy Reasoning
            logger.info("Step 4: Performing policy reasoning...")
            reasoning_response = self.agents["policy_reasoning"].process({
                "parsed_query": parse_response.data,
                "kg_analysis": kg_response.data
            })
            if not reasoning_response.success:
                return {"error": f"Policy reasoning failed: {reasoning_response.error_message}"}
            
            workflow_data["policy_reasoning"] = reasoning_response.data
            logger.info(f"Policy reasoning complete: {reasoning_response.data.get('eligibility_status', 'unknown')}")
            
            # Inter-agent delay
            time.sleep(self.inter_agent_delay)
            
            # Step 5: Financial Calculation
            logger.info("Step 5: Calculating financial amounts...")
            financial_response = self.agents["financial_calc"].process({
                "parsed_query": parse_response.data,
                "policy_reasoning": reasoning_response.data,
                "kg_analysis": kg_response.data
            })
            if not financial_response.success:
                return {"error": f"Financial calculation failed: {financial_response.error_message}"}
            
            workflow_data["financial_calculation"] = financial_response.data
            logger.info(f"Financial calculation complete: {financial_response.data.get('final_payout', 'N/A')}")
            
            # Inter-agent delay
            time.sleep(self.inter_agent_delay)
            
            # Step 6: Decision Synthesis
            logger.info("Step 6: Synthesizing final decision...")
            synthesis_response = self.agents["decision_synthesis"].process(workflow_data)
            if not synthesis_response.success:
                return {"error": f"Decision synthesis failed: {synthesis_response.error_message}"}
            
            logger.info(f"Final decision: {synthesis_response.data.get('Decision', 'unknown')}")
            return synthesis_response.data
            
        except Exception as e:
            logger.error(f"Error in multi-agent processing: {e}")
            return {"error": f"Multi-agent processing failed: {str(e)}"}


def create_multi_agent_orchestrator(kg_manager: EnhancedKnowledgeGraphManager) -> MultiAgentOrchestrator:
    """Factory function to create multi-agent orchestrator"""
    return MultiAgentOrchestrator(kg_manager)