"""
Gemini-based Multi-Agent System for Hybrid RAG
Replaces Anthropic Claude with Google Gemini models
"""

import json
import logging
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Google Gemini
import google.generativeai as genai

# Environment and utilities
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

class QueryIntent(str, Enum):
    """Enum for query intents"""
    CHECK_COVERAGE = "check_coverage"
    CALCULATE_PAYOUT = "calculate_payout"
    VERIFY_ELIGIBILITY = "verify_eligibility"
    EXPLAIN_POLICY = "explain_policy"
    COMPARE_OPTIONS = "compare_options"

class DecisionType(str, Enum):
    """Enum for decision types"""
    APPROVED = "Approved"
    REJECTED = "Rejected"
    REQUIRES_REVIEW = "Requires Further Review"

@dataclass
class AgentConfig:
    """Configuration for Gemini agents"""
    gemini_api_key: str = os.getenv("GEMINI_API_KEY")
    model_name: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
    temperature: float = 0.1
    max_tokens: int = 4000
    max_retries: int = 3

class GeminiAgent:
    """Base class for Gemini-powered agents"""
    
    def __init__(self, agent_name: str, config: AgentConfig = None):
        self.agent_name = agent_name
        self.config = config or AgentConfig()
        
        if not self.config.gemini_api_key:
            logger.warning(f"{agent_name}: GEMINI_API_KEY not found")
            self.llm = None
        else:
            try:
                genai.configure(api_key=self.config.gemini_api_key)
                self.llm = genai.GenerativeModel(
                    model_name=self.config.model_name,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_tokens
                    )
                )
                logger.info(f"{agent_name} initialized with Gemini")
            except Exception as e:
                logger.error(f"Failed to initialize {agent_name}: {e}")
                self.llm = None
    
    def _execute_with_retry(self, prompt: str, max_retries: int = None) -> str:
        """Execute prompt with retry logic"""
        max_retries = max_retries or self.config.max_retries
        
        for attempt in range(max_retries):
            try:
                if not self.llm:
                    raise Exception("LLM not initialized")
                
                response = self.llm.generate_content(prompt)
                if response.text:
                    logger.debug(f"{self.agent_name}: Success on attempt {attempt + 1}")
                    return response.text
                else:
                    raise Exception("Empty response from Gemini")
                    
            except Exception as e:
                logger.warning(f"{self.agent_name}: Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"{self.agent_name}: All attempts failed")
                    raise e
        
        return ""

class QueryParsingAgent(GeminiAgent):
    """
    Query parsing agent using Gemini for intent recognition and entity extraction
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("QueryParsingAgent", config)
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query into structured format"""
        if not self.llm:
            return self._fallback_parsing(query)
        
        try:
            logger.info(f"Parsing query: {query[:100]}...")
            
            prompt = f"""
            You are an expert query parsing agent for insurance policy analysis.
            
            Parse this insurance query and extract structured information:
            
            QUERY: "{query}"
            
            Extract the following information and respond in JSON format:
            {{
                "intent": "one of: check_coverage, calculate_payout, verify_eligibility, explain_policy, compare_options",
                "entities": {{
                    "age": "extracted age as number or null",
                    "procedure": "medical procedure name or null", 
                    "location": "city/location or null",
                    "amount": "monetary amount or null",
                    "duration": "policy duration or null",
                    "condition": "medical condition or null"
                }},
                "confidence": "confidence score 0-1",
                "ambiguities": ["list of unclear aspects"]
            }}
            
            Focus on extracting specific, actionable information from the query.
            """
            
            response = self._execute_with_retry(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response)
                logger.info(f"Query parsed successfully with intent: {result.get('intent')}")
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
                else:
                    raise Exception("Could not parse JSON from response")
            
        except Exception as e:
            logger.error(f"Error in query parsing: {e}")
            return self._fallback_parsing(query)
    
    def _fallback_parsing(self, query: str) -> Dict[str, Any]:
        """Fallback parsing without LLM"""
        logger.warning("Using fallback query parsing")
        
        entities = {}
        ambiguities = ["LLM not available - using simple parsing"]
        
        # Extract basic entities using regex
        age_match = re.search(r'(\d+)[-\s]?year[-\s]?old|age\s+(\d+)', query.lower())
        if age_match:
            entities['age'] = int(age_match.group(1) or age_match.group(2))
        
        # Extract amount
        amount_match = re.search(r'₹\s*(\d+(?:,\d+)*)', query)
        if amount_match:
            entities['amount'] = amount_match.group(1).replace(',', '')
        
        # Determine intent based on keywords
        intent = "check_coverage"  # Default
        if any(word in query.lower() for word in ['calculate', 'amount', 'cost', 'payout']):
            intent = "calculate_payout"
        elif any(word in query.lower() for word in ['eligible', 'qualify']):
            intent = "verify_eligibility"
        
        return {
            "intent": intent,
            "entities": entities,
            "confidence": 0.5,
            "ambiguities": ambiguities
        }

class PolicyReasoningAgent(GeminiAgent):
    """
    Policy reasoning agent using Gemini for complex policy analysis
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("PolicyReasoningAgent", config)
    
    def reason_about_policy(self, query_data: Dict[str, Any], context_data: List[Dict]) -> Dict[str, Any]:
        """Perform comprehensive policy reasoning"""
        if not self.llm:
            return self._fallback_reasoning(query_data, context_data)
        
        try:
            logger.info("Starting policy reasoning analysis")
            
            # Prepare context
            context_text = ""
            for i, ctx in enumerate(context_data[:5]):  # Limit context
                if isinstance(ctx, dict):
                    content = ctx.get('content', str(ctx))
                else:
                    content = str(ctx)
                context_text += f"Context {i+1}: {content}\n\n"
            
            prompt = f"""
            You are an expert insurance policy reasoning agent.
            
            Analyze this insurance query using the provided context and determine eligibility and coverage.
            
            QUERY DATA: {json.dumps(query_data, indent=2)}
            
            CONTEXT INFORMATION:
            {context_text}
            
            Perform step-by-step analysis:
            
            1. ELIGIBILITY ANALYSIS: Check if the person/situation meets policy requirements
            2. COVERAGE ANALYSIS: Determine what coverage applies and at what level
            3. RULE APPLICATION: Apply relevant policy rules and conditions
            4. CONFLICT RESOLUTION: Identify and resolve any conflicting information
            5. CONFIDENCE ASSESSMENT: Rate your confidence in the analysis
            
            Respond in JSON format:
            {{
                "eligibility_checks": {{
                    "age_eligible": true/false,
                    "location_covered": true/false,
                    "waiting_period_met": true/false,
                    "procedure_covered": true/false
                }},
                "coverage_analysis": {{
                    "coverage_percentage": "percentage as number",
                    "coverage_limit": "maximum amount covered",
                    "applicable_conditions": ["list of conditions"]
                }},
                "reasoning_steps": ["step 1", "step 2", "step 3"],
                "conflicts_identified": ["any conflicts found"],
                "confidence_score": 0.9
            }}
            """
            
            response = self._execute_with_retry(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response)
                logger.info(f"Policy reasoning completed with confidence: {result.get('confidence_score', 0)}")
                return result
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise Exception("Could not parse JSON from response")
            
        except Exception as e:
            logger.error(f"Error in policy reasoning: {e}")
            return self._fallback_reasoning(query_data, context_data)
    
    def _fallback_reasoning(self, query_data: Dict[str, Any], context_data: List) -> Dict[str, Any]:
        """Fallback reasoning without LLM"""
        logger.warning("Using fallback policy reasoning")
        
        # Simple rule-based reasoning
        eligibility_checks = {}
        reasoning_steps = ["Fallback reasoning without LLM"]
        
        # Basic age check
        entities = query_data.get('entities', {})
        if 'age' in entities:
            age = entities['age']
            eligibility_checks['age_eligible'] = 18 <= age <= 75
            reasoning_steps.append(f"Age {age} check: {'eligible' if eligibility_checks['age_eligible'] else 'not eligible'}")
        
        # Basic coverage assumption
        coverage_analysis = {
            "coverage_percentage": 80,
            "coverage_limit": "500000",
            "applicable_conditions": ["Standard policy conditions apply"]
        }
        
        return {
            "eligibility_checks": eligibility_checks,
            "coverage_analysis": coverage_analysis,
            "reasoning_steps": reasoning_steps,
            "conflicts_identified": ["Cannot detect conflicts without LLM"],
            "confidence_score": 0.3
        }

class FinancialCalculationAgent(GeminiAgent):
    """
    Financial calculation agent using Gemini for precise amount calculations
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("FinancialCalculationAgent", config)
    
    def calculate_amount(self, query_data: Dict[str, Any], reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate precise financial amounts"""
        if not self.llm:
            return self._fallback_calculation(query_data, reasoning_result)
        
        try:
            logger.info("Starting financial calculations")
            
            prompt = f"""
            You are an expert financial calculation agent for insurance claims.
            
            Calculate the precise payout amount based on the query and policy analysis.
            
            QUERY DATA: {json.dumps(query_data, indent=2)}
            POLICY ANALYSIS: {json.dumps(reasoning_result, indent=2)}
            
            Perform detailed calculation:
            
            1. IDENTIFY BASE AMOUNT: Determine the base calculation amount
            2. APPLY COVERAGE PERCENTAGE: Apply the coverage percentage from policy analysis  
            3. APPLY LIMITS: Check against policy limits and caps
            4. APPLY DEDUCTIBLES: Subtract any applicable deductibles
            5. FINAL VERIFICATION: Verify the calculation is correct
            
            Respond in JSON format:
            {{
                "calculated_amount": 450000,
                "calculation_steps": [
                    "Base amount: ₹500,000",
                    "Coverage: 90% = ₹450,000", 
                    "Applied policy limit check",
                    "Final amount: ₹450,000"
                ],
                "applied_rules": ["90% coverage rule", "Maximum limit ₹5,00,000"],
                "assumptions": ["Standard deductible waived"]
            }}
            
            Ensure all amounts are in rupees and calculations are precise.
            """
            
            response = self._execute_with_retry(prompt)
            
            try:
                result = json.loads(response)
                logger.info(f"Financial calculation completed: ₹{result.get('calculated_amount', 0)}")
                return result
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise Exception("Could not parse JSON from response")
            
        except Exception as e:
            logger.error(f"Error in financial calculation: {e}")
            return self._fallback_calculation(query_data, reasoning_result)
    
    def _fallback_calculation(self, query_data: Dict[str, Any], reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback calculation without LLM"""
        logger.warning("Using fallback financial calculation")
        
        # Simple calculation logic
        entities = query_data.get('entities', {})
        coverage_analysis = reasoning_result.get('coverage_analysis', {})
        
        base_amount = 100000  # Default
        if 'amount' in entities:
            base_amount = float(entities['amount'])
        
        coverage_percentage = coverage_analysis.get('coverage_percentage', 80)
        calculated_amount = base_amount * (coverage_percentage / 100)
        
        return {
            "calculated_amount": int(calculated_amount),
            "calculation_steps": [
                f"Base amount: ₹{base_amount:,.0f}",
                f"Coverage: {coverage_percentage}% = ₹{calculated_amount:,.0f}"
            ],
            "applied_rules": [f"{coverage_percentage}% coverage rule"],
            "assumptions": ["Default calculation without LLM"]
        }

class DecisionSynthesisAgent(GeminiAgent):
    """
    Decision synthesis agent using Gemini for final decision making
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("DecisionSynthesisAgent", config)
    
    def synthesize_decision(self, query_data: Dict[str, Any], reasoning_result: Dict[str, Any], 
                          calculation_result: Dict[str, Any], context_data: List) -> Dict[str, Any]:
        """Synthesize final decision from all agent outputs"""
        if not self.llm:
            return self._fallback_decision(query_data, reasoning_result, calculation_result)
        
        try:
            logger.info("Starting final decision synthesis")
            
            # Prepare relevant clauses from context
            relevant_clauses = []
            for i, ctx in enumerate(context_data[:3]):
                if isinstance(ctx, dict):
                    content = ctx.get('content', str(ctx))
                    metadata = ctx.get('metadata', {})
                else:
                    content = str(ctx)
                    metadata = {}
                
                relevant_clauses.append({
                    "clause_text": content[:500] + "..." if len(content) > 500 else content,
                    "document_id": metadata.get('document_id', 'unknown'),
                    "page_section": metadata.get('page_section', f'section_{i}'),
                    "retrieval_source": metadata.get('source', 'context')
                })
            
            prompt = f"""
            You are the final decision synthesis agent for insurance claim processing.
            
            Synthesize all the analysis into a final authoritative decision.
            
            QUERY DATA: {json.dumps(query_data, indent=2)}
            REASONING RESULT: {json.dumps(reasoning_result, indent=2)}  
            CALCULATION RESULT: {json.dumps(calculation_result, indent=2)}
            
            Based on this analysis, make the final decision:
            
            1. DECISION: "Approved", "Rejected", or "Requires Further Review"
            2. AMOUNT: The final payout amount (as string) or "N/A"
            3. JUSTIFICATION: Comprehensive explanation of the decision
            
            Decision Criteria:
            - Approved: All eligibility met, coverage applies, amount calculated
            - Rejected: Clear policy violations or exclusions
            - Requires Further Review: Ambiguous cases or missing information
            
            Respond in JSON format:
            {{
                "Decision": "Approved",
                "Amount": "450000", 
                "Justification": "Detailed explanation of decision with policy references and reasoning"
            }}
            
            Make the justification comprehensive and reference specific policy aspects.
            """
            
            response = self._execute_with_retry(prompt)
            
            try:
                result = json.loads(response)
                
                # Add relevant clauses to the result
                result["Relevant_Clauses"] = relevant_clauses
                
                logger.info(f"Final decision synthesized: {result.get('Decision')}")
                return result
                
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    result["Relevant_Clauses"] = relevant_clauses
                    return result
                else:
                    raise Exception("Could not parse JSON from response")
            
        except Exception as e:
            logger.error(f"Error in decision synthesis: {e}")
            return self._fallback_decision(query_data, reasoning_result, calculation_result)
    
    def _fallback_decision(self, query_data: Dict[str, Any], reasoning_result: Dict[str, Any], 
                          calculation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback decision without LLM"""
        logger.warning("Using fallback decision synthesis")
        
        # Simple decision logic
        eligibility_checks = reasoning_result.get('eligibility_checks', {})
        calculated_amount = calculation_result.get('calculated_amount', 0)
        
        # Check if generally eligible
        eligible_count = sum(1 for check in eligibility_checks.values() if check)
        total_checks = len(eligibility_checks)
        
        if total_checks > 0 and eligible_count == total_checks:
            decision = "Approved"
            amount = str(calculated_amount) if calculated_amount > 0 else "N/A"
        elif eligible_count == 0:
            decision = "Rejected" 
            amount = "N/A"
        else:
            decision = "Requires Further Review"
            amount = "N/A"
        
        justification = f"Decision based on eligibility assessment. {eligible_count}/{total_checks} checks passed. "
        if calculated_amount > 0:
            justification += f"Calculated amount: ₹{calculated_amount:,}. "
        justification += "Limited analysis due to LLM unavailability."
        
        return {
            "Decision": decision,
            "Amount": amount,
            "Justification": justification,
            "Relevant_Clauses": [{
                "clause_text": "Fallback processing - limited information available",
                "document_id": "system_fallback",
                "page_section": "N/A",
                "retrieval_source": "System"
            }]
        }

class GeminiAgentSystem:
    """
    Orchestrates Gemini-based multi-agent workflow
    """
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        
        # Initialize agents
        self.query_parser = QueryParsingAgent(config)
        self.policy_reasoner = PolicyReasoningAgent(config)
        self.financial_calculator = FinancialCalculationAgent(config)
        self.decision_synthesizer = DecisionSynthesisAgent(config)
        
        logger.info("Gemini Agent System initialized")
    
    def process_query(self, query: str, context_data: List = None) -> Dict[str, Any]:
        """Process query through complete multi-agent pipeline"""
        context_data = context_data or []
        
        logger.info(f"Starting Gemini multi-agent processing for query: {query[:100]}...")
        
        try:
            # Step 1: Parse query
            logger.info("Step 1: Query Parsing")  
            parsed_query = self.query_parser.parse_query(query)
            
            # Step 2: Policy reasoning
            logger.info("Step 2: Policy Reasoning")
            reasoning_result = self.policy_reasoner.reason_about_policy(parsed_query, context_data)
            
            # Step 3: Financial calculation
            logger.info("Step 3: Financial Calculation")
            calculation_result = self.financial_calculator.calculate_amount(parsed_query, reasoning_result)
            
            # Step 4: Decision synthesis
            logger.info("Step 4: Decision Synthesis")
            final_decision = self.decision_synthesizer.synthesize_decision(
                parsed_query, reasoning_result, calculation_result, context_data
            )
            
            # Add processing metadata
            final_decision["Processing_Info"] = {
                "processing_method": "gemini_multi_agent",
                "kg_mode": "fallback",
                "documents_available": len(context_data),
                "model_used": self.config.model_name
            }
            
            logger.info(f"Gemini multi-agent processing completed: {final_decision.get('Decision')}")
            return final_decision
            
        except Exception as e:
            logger.error(f"Error in multi-agent processing: {e}")
            
            # Return error decision
            return {
                "Decision": "Requires Further Review",
                "Amount": "N/A", 
                "Justification": f"Processing error occurred: {str(e)}. Please try again or contact support.",
                "Relevant_Clauses": [{
                    "clause_text": "System error during processing",
                    "document_id": "system_error",
                    "page_section": "N/A",
                    "retrieval_source": "System"
                }],
                "Processing_Info": {
                    "processing_method": "error_fallback",
                    "error": str(e)
                }
            }

# Factory function
def create_agent_system() -> GeminiAgentSystem:
    """Create and return a Gemini agent system"""
    return GeminiAgentSystem()

# Global instance
_agent_system = None

def get_agent_system() -> GeminiAgentSystem:
    """Get global agent system instance"""
    global _agent_system
    if _agent_system is None:
        _agent_system = GeminiAgentSystem()
    return _agent_system