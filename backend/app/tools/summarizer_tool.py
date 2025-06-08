"""
Summarizer Tool for the Legal Intelligence Platform.

This tool provides abstractive LLM-based summaries with templated outputs
for various legal document types and analysis purposes.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import re

# LangChain imports
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain.schema import BaseMessage
from pydantic import BaseModel, Field

# Custom imports
from app.euri_client import euri_chat_completion
from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class SummaryType(Enum):
    """Types of summaries that can be generated."""
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    RISK_FOCUSED = "risk_focused"
    FINANCIAL = "financial"
    COMPLIANCE = "compliance"
    CLAUSE_BY_CLAUSE = "clause_by_clause"
    TIMELINE = "timeline"
    PARTIES_OBLIGATIONS = "parties_obligations"


class DocumentType(Enum):
    """Document types for template selection."""
    CONTRACT = "contract"
    AGREEMENT = "agreement"
    POLICY = "policy"
    REGULATION = "regulation"
    CASE_LAW = "case_law"
    STATUTE = "statute"
    MEMO = "memo"
    BRIEF = "brief"


@dataclass
class SummaryTemplate:
    """Template for generating structured summaries."""
    name: str
    description: str
    sections: List[str]
    prompt_template: str
    output_format: str
    max_length: int
    target_audience: str


class SummaryResult(BaseModel):
    """Pydantic model for summary results."""
    summary_type: str = Field(description="Type of summary generated")
    document_type: str = Field(description="Type of document summarized")
    executive_summary: str = Field(description="High-level executive summary")
    key_points: List[str] = Field(description="Key points and findings")
    sections: Dict[str, str] = Field(description="Detailed sections")
    metadata: Dict[str, Any] = Field(description="Summary metadata")
    confidence_score: float = Field(description="Confidence in summary quality")


class SummarizerTool(BaseTool):
    """
    Advanced Summarizer Tool for legal documents.
    
    This tool provides:
    - Abstractive LLM-based summaries
    - Multiple summary templates for different purposes
    - Document-type specific formatting
    - Structured output with key sections
    - Confidence scoring for summary quality
    - Customizable length and detail levels
    """
    
    name = "summarizer"
    description = "Generate abstractive summaries of legal documents with templated outputs"
    
    def __init__(self):
        """Initialize the Summarizer Tool."""
        super().__init__()
        
        # Initialize summary templates
        self.templates = self._initialize_templates()
        
        # Output parser
        self.output_parser = PydanticOutputParser(pydantic_object=SummaryResult)
    
    def _initialize_templates(self) -> Dict[str, SummaryTemplate]:
        """Initialize summary templates for different document types and purposes."""
        templates = {}
        
        # Executive Summary Template
        templates["executive"] = SummaryTemplate(
            name="Executive Summary",
            description="High-level summary for executives and decision makers",
            sections=["overview", "key_terms", "risks", "recommendations"],
            prompt_template="""
            Create an executive summary of the following legal document:
            
            Document: {document_text}
            
            Provide a concise executive summary that includes:
            1. Document Overview (2-3 sentences)
            2. Key Terms and Conditions (bullet points)
            3. Primary Risks and Concerns (bullet points)
            4. Strategic Recommendations (bullet points)
            
            Keep the summary under {max_length} words and focus on business impact.
            """,
            output_format="structured",
            max_length=500,
            target_audience="executives"
        )
        
        # Technical Legal Summary Template
        templates["technical"] = SummaryTemplate(
            name="Technical Legal Summary",
            description="Detailed technical analysis for legal professionals",
            sections=["legal_analysis", "clauses", "precedents", "compliance"],
            prompt_template="""
            Provide a technical legal analysis of the following document:
            
            Document: {document_text}
            
            Include:
            1. Legal Framework and Jurisdiction
            2. Clause-by-Clause Analysis
            3. Legal Precedents and References
            4. Compliance Requirements
            5. Potential Legal Issues
            
            Maximum length: {max_length} words. Use legal terminology appropriately.
            """,
            output_format="detailed",
            max_length=1500,
            target_audience="legal_professionals"
        )
        
        # Risk-Focused Summary Template
        templates["risk_focused"] = SummaryTemplate(
            name="Risk Assessment Summary",
            description="Summary focused on identifying and analyzing risks",
            sections=["risk_overview", "financial_risks", "legal_risks", "operational_risks", "mitigation"],
            prompt_template="""
            Analyze the following document for risks and provide a risk-focused summary:
            
            Document: {document_text}
            
            Identify and analyze:
            1. Overall Risk Profile
            2. Financial Risks and Exposure
            3. Legal and Compliance Risks
            4. Operational Risks
            5. Risk Mitigation Strategies
            
            Categorize risks as HIGH, MEDIUM, or LOW. Maximum {max_length} words.
            """,
            output_format="risk_matrix",
            max_length=800,
            target_audience="risk_managers"
        )
        
        # Financial Summary Template
        templates["financial"] = SummaryTemplate(
            name="Financial Terms Summary",
            description="Summary focused on financial terms and obligations",
            sections=["financial_overview", "payment_terms", "costs", "penalties", "financial_risks"],
            prompt_template="""
            Extract and summarize all financial terms from the following document:
            
            Document: {document_text}
            
            Provide:
            1. Financial Overview
            2. Payment Terms and Schedule
            3. Costs and Fees
            4. Penalties and Late Charges
            5. Financial Risk Assessment
            
            Include specific amounts, dates, and conditions. Maximum {max_length} words.
            """,
            output_format="financial_breakdown",
            max_length=600,
            target_audience="financial_analysts"
        )
        
        # Compliance Summary Template
        templates["compliance"] = SummaryTemplate(
            name="Compliance Summary",
            description="Summary focused on regulatory compliance requirements",
            sections=["compliance_overview", "regulations", "requirements", "obligations", "monitoring"],
            prompt_template="""
            Analyze the following document for compliance requirements:
            
            Document: {document_text}
            
            Identify:
            1. Applicable Regulations and Standards
            2. Compliance Requirements
            3. Reporting Obligations
            4. Monitoring and Audit Requirements
            5. Non-Compliance Consequences
            
            Focus on actionable compliance items. Maximum {max_length} words.
            """,
            output_format="compliance_checklist",
            max_length=700,
            target_audience="compliance_officers"
        )
        
        # Timeline Summary Template
        templates["timeline"] = SummaryTemplate(
            name="Timeline and Deadlines Summary",
            description="Summary focused on important dates and deadlines",
            sections=["timeline_overview", "key_dates", "deadlines", "milestones", "renewal_dates"],
            prompt_template="""
            Extract all important dates, deadlines, and timeline information from:
            
            Document: {document_text}
            
            Create a timeline summary including:
            1. Contract/Document Timeline
            2. Key Dates and Deadlines
            3. Performance Milestones
            4. Renewal and Termination Dates
            5. Notice Requirements
            
            Present in chronological order. Maximum {max_length} words.
            """,
            output_format="timeline",
            max_length=500,
            target_audience="project_managers"
        )
        
        return templates
    
    def _run(self, document_text: str, **kwargs) -> str:
        """
        Run the summarizer tool.
        
        Args:
            document_text (str): Text to summarize
            **kwargs: Additional arguments
            
        Returns:
            str: Generated summary
        """
        try:
            summary_type = kwargs.get('summary_type', 'executive')
            document_type = kwargs.get('document_type', 'contract')
            
            result = self.generate_summary(
                document_text=document_text,
                summary_type=summary_type,
                document_type=document_type,
                **kwargs
            )
            
            # Format result as string
            formatted_summary = self._format_summary_output(result)
            return formatted_summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return f"Summarization failed: {str(e)}"
    
    def generate_summary(self,
                        document_text: str,
                        summary_type: str = "executive",
                        document_type: str = "contract",
                        custom_instructions: Optional[str] = None,
                        max_length: Optional[int] = None) -> SummaryResult:
        """
        Generate a structured summary of the document.
        
        Args:
            document_text (str): Text to summarize
            summary_type (str): Type of summary to generate
            document_type (str): Type of document being summarized
            custom_instructions (str, optional): Custom instructions for summarization
            max_length (int, optional): Maximum length override
            
        Returns:
            SummaryResult: Structured summary result
        """
        try:
            # Get appropriate template
            template = self.templates.get(summary_type, self.templates["executive"])
            
            # Override max length if specified
            if max_length:
                template.max_length = max_length
            
            # Prepare prompt
            prompt = self._prepare_prompt(document_text, template, custom_instructions)
            
            # Generate summary using EURI
            summary_content = self._generate_with_euri(prompt, template)
            
            # Parse and structure the result
            result = self._parse_summary_result(
                summary_content, summary_type, document_type, template
            )
            
            # Calculate confidence score
            result.confidence_score = self._calculate_confidence_score(
                document_text, result, template
            )
            
            logger.info(f"Generated {summary_type} summary for {document_type}")
            return result
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Return empty result on failure
            return SummaryResult(
                summary_type=summary_type,
                document_type=document_type,
                executive_summary="Summary generation failed",
                key_points=[],
                sections={},
                metadata={"error": str(e)},
                confidence_score=0.0
            )
    
    def _prepare_prompt(self,
                       document_text: str,
                       template: SummaryTemplate,
                       custom_instructions: Optional[str] = None) -> str:
        """Prepare the prompt for summary generation."""
        
        # Truncate document if too long (keep first and last parts)
        max_doc_length = 8000  # Adjust based on model limits
        if len(document_text) > max_doc_length:
            mid_point = max_doc_length // 2
            truncated_text = (
                document_text[:mid_point] + 
                "\n\n[... document truncated ...]\n\n" + 
                document_text[-mid_point:]
            )
        else:
            truncated_text = document_text
        
        # Format the prompt
        prompt = template.prompt_template.format(
            document_text=truncated_text,
            max_length=template.max_length
        )
        
        # Add custom instructions if provided
        if custom_instructions:
            prompt += f"\n\nAdditional Instructions: {custom_instructions}"
        
        # Add output format instructions
        prompt += f"""
        
        Please provide the response in the following JSON format:
        {{
            "executive_summary": "Brief overview of the document",
            "key_points": ["point 1", "point 2", "point 3"],
            "sections": {{
                "section_name": "section_content"
            }},
            "metadata": {{
                "word_count": number,
                "complexity": "low/medium/high",
                "document_structure": "description"
            }}
        }}
        """
        
        return prompt
    
    def _generate_with_euri(self, prompt: str, template: SummaryTemplate) -> str:
        """Generate summary using EURI API."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"You are a legal document summarization expert. Generate {template.description.lower()} for {template.target_audience}. Be accurate, concise, and professional."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = euri_chat_completion(
                messages=messages,
                model=settings.euri_model,
                temperature=0.3,  # Lower temperature for more consistent summaries
                max_tokens=min(template.max_length * 2, 2000)  # Allow some buffer
            )
            
            return response
            
        except Exception as e:
            logger.error(f"EURI API call failed: {e}")
            raise
    
    def _parse_summary_result(self,
                             summary_content: str,
                             summary_type: str,
                             document_type: str,
                             template: SummaryTemplate) -> SummaryResult:
        """Parse the generated summary into structured result."""
        try:
            # Try to parse as JSON first
            try:
                parsed_json = json.loads(summary_content)
                
                return SummaryResult(
                    summary_type=summary_type,
                    document_type=document_type,
                    executive_summary=parsed_json.get("executive_summary", ""),
                    key_points=parsed_json.get("key_points", []),
                    sections=parsed_json.get("sections", {}),
                    metadata=parsed_json.get("metadata", {}),
                    confidence_score=0.0  # Will be calculated separately
                )
                
            except json.JSONDecodeError:
                # Fallback to text parsing
                return self._parse_text_summary(summary_content, summary_type, document_type)
                
        except Exception as e:
            logger.error(f"Summary parsing failed: {e}")
            # Return basic result
            return SummaryResult(
                summary_type=summary_type,
                document_type=document_type,
                executive_summary=summary_content[:500],
                key_points=[],
                sections={"content": summary_content},
                metadata={"parsing_method": "fallback"},
                confidence_score=0.5
            )
    
    def _parse_text_summary(self, content: str, summary_type: str, document_type: str) -> SummaryResult:
        """Parse unstructured text summary."""
        lines = content.split('\n')
        
        # Extract executive summary (first paragraph)
        executive_summary = ""
        key_points = []
        sections = {}
        
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            if any(keyword in line.lower() for keyword in ['overview', 'summary', 'key', 'risk', 'recommendation']):
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                current_section = line
                current_content = []
            
            # Check if line is a bullet point
            elif line.startswith(('-', '•', '*', '1.', '2.', '3.')):
                key_points.append(line.lstrip('-•*123456789. '))
                current_content.append(line)
            
            else:
                if not executive_summary and len(line) > 50:
                    executive_summary = line
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return SummaryResult(
            summary_type=summary_type,
            document_type=document_type,
            executive_summary=executive_summary or content[:200],
            key_points=key_points[:10],  # Limit to 10 key points
            sections=sections,
            metadata={"parsing_method": "text_parsing"},
            confidence_score=0.7
        )
    
    def _calculate_confidence_score(self,
                                   original_text: str,
                                   summary_result: SummaryResult,
                                   template: SummaryTemplate) -> float:
        """Calculate confidence score for the summary quality."""
        try:
            score = 0.8  # Base score
            
            # Check if summary has required sections
            if summary_result.sections:
                score += 0.1
            
            # Check if key points are extracted
            if summary_result.key_points:
                score += 0.1
            
            # Check length appropriateness
            summary_length = len(summary_result.executive_summary)
            if 50 <= summary_length <= template.max_length:
                score += 0.1
            else:
                score -= 0.1
            
            # Check for specific content based on template
            content_lower = summary_result.executive_summary.lower()
            if template.name == "Risk Assessment Summary":
                if any(risk_word in content_lower for risk_word in ['risk', 'liability', 'exposure']):
                    score += 0.1
            elif template.name == "Financial Terms Summary":
                if any(fin_word in content_lower for fin_word in ['payment', 'cost', 'fee', 'amount']):
                    score += 0.1
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _format_summary_output(self, result: SummaryResult) -> str:
        """Format summary result as readable text."""
        output = []
        
        output.append(f"# {result.summary_type.title()} Summary")
        output.append(f"Document Type: {result.document_type.title()}")
        output.append(f"Confidence Score: {result.confidence_score:.2f}")
        output.append("")
        
        # Executive Summary
        output.append("## Executive Summary")
        output.append(result.executive_summary)
        output.append("")
        
        # Key Points
        if result.key_points:
            output.append("## Key Points")
            for point in result.key_points:
                output.append(f"• {point}")
            output.append("")
        
        # Detailed Sections
        if result.sections:
            output.append("## Detailed Analysis")
            for section_name, section_content in result.sections.items():
                output.append(f"### {section_name.title()}")
                output.append(section_content)
                output.append("")
        
        return "\n".join(output)
    
    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available summary templates.
        
        Returns:
            Dict: Template information
        """
        template_info = {}
        
        for key, template in self.templates.items():
            template_info[key] = {
                "name": template.name,
                "description": template.description,
                "sections": template.sections,
                "max_length": template.max_length,
                "target_audience": template.target_audience,
                "output_format": template.output_format
            }
        
        return template_info
