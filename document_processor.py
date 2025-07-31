"""
Enhanced Document Processor with PDF Support
Handles both text files and PDF documents for Knowledge Graph ingestion
"""

import os
import logging
import time
import random
from typing import Dict, List, Any, Optional
import PyPDF2
import pdfplumber
from pathlib import Path
import google.generativeai as genai
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Enhanced document processor that handles multiple file formats
    including PDF files and extracts structured information using AI
    """
    
    def __init__(self, gemini_api_key: str, base_delay: float = 5.0):
        """
        Initialize document processor with Gemini API and rate limiting
        
        Args:
            gemini_api_key: Google Gemini API key
            base_delay: Base delay between API calls for rate limiting
        """
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.base_delay = base_delay
        self.last_api_call = 0
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file using multiple methods for best results
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        text_content = ""
        
        try:
            # Method 1: Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            
            logger.info(f"Successfully extracted {len(text_content)} characters using pdfplumber")
            
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyPDF2...")
            
            try:
                # Method 2: Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() + "\n"
                
                logger.info(f"Successfully extracted {len(text_content)} characters using PyPDF2")
                
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed: {e2}")
                return ""
        
        return text_content.strip()
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from any supported file format
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension in ['.txt', '.md']:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}")
                return ""
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return ""
    
    def extract_entities_with_ai(self, text_content: str, document_type: str = "insurance_policy", max_retries: int = 3) -> Dict[str, Any]:
        """
        Use Gemini AI to extract structured entities from document text with rate limiting
        
        Args:
            text_content: Raw text content from document
            document_type: Type of document for context
            max_retries: Maximum number of retry attempts
            
        Returns:
            Structured entities and relationships
        """
        for attempt in range(max_retries):
            try:
                # Rate limiting: ensure minimum delay between API calls
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call
                
                if time_since_last_call < self.base_delay:
                    sleep_time = self.base_delay - time_since_last_call
                    # Add jitter to avoid thundering herd
                    sleep_time += random.uniform(1, 3)
                    logger.info(f"DocumentProcessor: Waiting {sleep_time:.2f}s to avoid rate limit")
                    time.sleep(sleep_time)
                
                logger.info(f"DocumentProcessor: Extracting entities (attempt {attempt + 1}/{max_retries})")
                
                extraction_prompt = f"""
            You are an expert information extraction system. Extract structured information from the following {document_type} document.
            
            Document Text:
            {text_content[:8000]}  # Limit to avoid token limits
            
            Extract and return the following information as JSON:
            {{
                "document_metadata": {{
                    "title": "document title if found",
                    "type": "policy type or document category",
                    "version": "version if mentioned",
                    "effective_date": "date if mentioned"
                }},
                "policies": [
                    {{
                        "id": "policy identifier",
                        "name": "policy name",
                        "type": "policy type",
                        "coverage_amount": "coverage amount as string",
                        "premium": "premium amount",
                        "term": "policy term"
                    }}
                ],
                "procedures": [
                    {{
                        "name": "procedure name",
                        "category": "medical category",
                        "coverage_limit": "coverage amount",
                        "waiting_period": "waiting period in days",
                        "conditions": "any specific conditions"
                    }}
                ],
                "eligibility_criteria": [
                    {{
                        "type": "eligibility type (age, location, etc.)",
                        "min_value": "minimum value if applicable",
                        "max_value": "maximum value if applicable",
                        "description": "detailed description",
                        "conditions": "additional conditions"
                    }}
                ],
                "financial_terms": [
                    {{
                        "term_type": "deductible, co-payment, etc.",
                        "amount": "amount",
                        "description": "description",
                        "applies_to": "what it applies to"
                    }}
                ],
                "geographic_coverage": [
                    {{
                        "location": "city or region name",
                        "coverage_type": "full, partial, emergency",
                        "special_conditions": "any special conditions"
                    }}
                ],
                "exclusions": [
                    {{
                        "type": "exclusion type",
                        "description": "what is excluded",
                        "conditions": "under what conditions"
                    }}
                ],
                "key_clauses": [
                    {{
                        "clause_id": "section reference",
                        "title": "clause title",
                        "content": "clause content",
                        "importance": "high, medium, low"
                    }}
                ]
            }}
            
            Be precise and extract only information that is explicitly mentioned in the document.
            Use "null" for missing information.
            Focus on insurance-related terms, medical procedures, coverage limits, and eligibility criteria.
            """
            
                response = self.model.generate_content(extraction_prompt)
                self.last_api_call = time.time()
                response_text = response.text
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        entities = json.loads(json_match.group())
                        logger.info("Successfully extracted entities using AI")
                        return entities
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing error: {e}")
                
                # If no JSON found, fallback
                logger.warning("No valid JSON found in AI response, using fallback")
                return self._fallback_entity_extraction(text_content)
                
            except Exception as e:
                error_message = str(e)
                logger.warning(f"DocumentProcessor attempt {attempt + 1} failed: {error_message}")
                
                # Check if it's a rate limit error
                if "rate limit" in error_message.lower() or "quota" in error_message.lower() or "429" in error_message:
                    if attempt < max_retries - 1:
                        # Exponential backoff for rate limit errors
                        backoff_time = (2 ** attempt) * self.base_delay + random.uniform(2, 5)
                        logger.info(f"DocumentProcessor: Rate limit hit, backing off for {backoff_time:.2f}s")
                        time.sleep(backoff_time)
                        continue
                    else:
                        logger.error(f"DocumentProcessor: Rate limit exceeded after {max_retries} attempts")
                        return self._fallback_entity_extraction(text_content)
                else:
                    # For non-rate-limit errors, try fallback immediately
                    logger.error(f"AI entity extraction failed: {e}")
                    return self._fallback_entity_extraction(text_content)
        
        # If all retries failed, use fallback
        logger.error(f"All {max_retries} attempts failed, using fallback extraction")
        return self._fallback_entity_extraction(text_content)
    
    def _fallback_entity_extraction(self, text_content: str) -> Dict[str, Any]:
        """Fallback entity extraction using pattern matching"""
        entities = {
            "document_metadata": {"title": "Unknown Document", "type": "insurance_policy"},
            "policies": [],
            "procedures": [],
            "eligibility_criteria": [],
            "financial_terms": [],
            "geographic_coverage": [],
            "exclusions": [],
            "key_clauses": []
        }
        
        # Basic pattern matching for common insurance terms
        procedure_patterns = [
            r"(?i)(surgery|operation|treatment|procedure)",
            r"(?i)(knee|heart|dental|eye|orthopedic)",
            r"(?i)coverage[:\s]*(?:rs\.?|rupees?)?[:\s]*(\d+(?:,\d+)*)",
            r"(?i)waiting period[:\s]*(\d+)\s*(day|month|year)"
        ]
        
        for pattern in procedure_patterns:
            matches = re.findall(pattern, text_content)
            for match in matches:
                if isinstance(match, tuple):
                    match_text = ' '.join(str(m) for m in match if m)
                else:
                    match_text = str(match)
                
                entities["procedures"].append({
                    "name": match_text.title(),
                    "category": "medical",
                    "coverage_limit": "unknown",
                    "waiting_period": "unknown",
                    "conditions": "as per policy terms"
                })
        
        # Extract financial terms
        financial_patterns = [
            r"(?i)(?:rs\.?|rupees?)[:\s]*(\d+(?:,\d+)*)",
            r"(?i)premium[:\s]*(?:rs\.?)?[:\s]*(\d+(?:,\d+)*)"
        ]
        
        for pattern in financial_patterns:
            matches = re.findall(pattern, text_content)
            for match in matches:
                entities["financial_terms"].append({
                    "term_type": "amount",
                    "amount": match,
                    "description": "Financial amount mentioned in document",
                    "applies_to": "general coverage"
                })
        
        logger.info("Used fallback entity extraction")
        return entities
    
    def process_document(self, file_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a complete document and extract all relevant information
        
        Args:
            file_path: Path to the document file
            document_id: Optional document identifier
            
        Returns:
            Complete processing results
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": f"File not found: {file_path}"}
        
        if not document_id:
            document_id = os.path.basename(file_path)
        
        logger.info(f"Processing document: {file_path}")
        
        # Extract text content
        text_content = self.extract_text_from_file(file_path)
        if not text_content:
            return {"error": f"Could not extract text from {file_path}"}
        
        # Extract structured entities using AI
        entities = self.extract_entities_with_ai(text_content)
        
        # Prepare final result
        result = {
            "document_id": document_id,
            "file_path": file_path,
            "text_content": text_content,
            "entities": entities,
            "processing_status": "success",
            "text_length": len(text_content)
        }
        
        logger.info(f"Successfully processed document: {document_id}")
        return result
    
    def process_documents_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Process all supported documents in a folder
        
        Args:
            folder_path: Path to folder containing documents
            
        Returns:
            List of processing results for each document
        """
        results = []
        supported_extensions = ['.pdf', '.txt', '.md']
        
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return results
        
        # Find all supported files
        for file_path in Path(folder_path).iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                result = self.process_document(str(file_path))
                results.append(result)
        
        logger.info(f"Processed {len(results)} documents from {folder_path}")
        return results


def create_document_processor(api_key: str) -> DocumentProcessor:
    """Factory function to create document processor"""
    return DocumentProcessor(api_key)