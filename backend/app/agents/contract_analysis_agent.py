"""
Contract Analysis Agent for the Legal Intelligence Platform.

This agent specializes in analyzing legal contracts using LangChain and LangGraph,
providing comprehensive contract review, clause analysis, and risk assessment.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# LangChain imports
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from langchain.schema import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# Custom imports
from app.euri_client import euri_chat_completion
from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ContractAnalysisResult:
    """Data class for contract analysis results."""
    contract_type: str
    parties: List[Dict[str, Any]]
    key_terms: Dict[str, Any]
    clauses: List[Dict[str, Any]]
    obligations: List[Dict[str, Any]]
    risks: List[Dict[str, Any]]
    important_dates: List[Dict[str, Any]]
    financial_terms: Dict[str, Any]
    compliance_issues: List[Dict[str, Any]]
    recommendations: List[str]
    summary: str
    confidence_score: float


class ContractParty(BaseModel):
    """Pydantic model for contract party."""
    name: str = Field(description="Name of the party")
    role: str = Field(description="Role in the contract (e.g., buyer, seller, contractor)")
    address: Optional[str] = Field(description="Address of the party")
    contact_info: Optional[str] = Field(description="Contact information")


class ContractClause(BaseModel):
    """Pydantic model for contract clause."""
    title: str = Field(description="Title or type of the clause")
    content: str = Field(description="Content of the clause")
    importance: str = Field(description="Importance level: high, medium, low")
    risk_level: str = Field(description="Risk level: high, medium, low")
    analysis: str = Field(description="Analysis of the clause")


class ContractTerms(BaseModel):
    """Pydantic model for contract terms."""
    duration: Optional[str] = Field(description="Contract duration")
    start_date: Optional[str] = Field(description="Contract start date")
    end_date: Optional[str] = Field(description="Contract end date")
    renewal_terms: Optional[str] = Field(description="Renewal terms")
    termination_conditions: Optional[str] = Field(description="Termination conditions")


class FinancialTerms(BaseModel):
    """Pydantic model for financial terms."""
    total_value: Optional[str] = Field(description="Total contract value")
    payment_schedule: Optional[str] = Field(description="Payment schedule")
    currency: Optional[str] = Field(description="Currency")
    penalties: Optional[str] = Field(description="Financial penalties")
    late_fees: Optional[str] = Field(description="Late payment fees")


class AgentState(TypedDict):
    """State for the contract analysis agent."""
    messages: List[BaseMessage]
    contract_text: str
    analysis_results: Optional[ContractAnalysisResult]
    current_step: str
    errors: List[str]
    extracted_data: Dict[str, Any]


class ContractAnalysisAgent:
    """
    Specialized agent for comprehensive contract analysis.
    
    This agent uses LangChain and LangGraph to perform detailed analysis of
    legal contracts including:
    - Contract type identification
    - Party extraction and analysis
    - Key terms and conditions analysis
    - Clause-by-clause review
    - Risk assessment
    - Compliance checking
    - Financial terms analysis
    - Important dates extraction
    """
    
    def __init__(self):
        """Initialize the contract analysis agent."""
        self.graph = self._build_analysis_graph()
        self.contract_types = [
            "employment", "service", "sales", "lease", "partnership",
            "licensing", "nda", "consulting", "construction", "supply"
        ]
        
    def _build_analysis_graph(self) -> StateGraph:
        """Build the LangGraph workflow for contract analysis."""
        
        def identify_contract_type(state: AgentState) -> AgentState:
            """Identify the type of contract."""
            try:
                contract_text = state["contract_text"]
                
                prompt = PromptTemplate(
                    input_variables=["contract_text", "contract_types"],
                    template="""
                    Analyze the following contract and identify its type.
                    
                    Contract Types: {contract_types}
                    
                    Contract Text:
                    {contract_text}
                    
                    Respond with just the contract type from the list above, or "other" if it doesn't match any.
                    """
                )
                
                messages = [
                    {"role": "system", "content": "You are a legal contract analysis expert."},
                    {"role": "user", "content": prompt.format(
                        contract_text=contract_text[:2000],  # Limit for efficiency
                        contract_types=", ".join(self.contract_types)
                    )}
                ]
                
                contract_type = euri_chat_completion(
                    messages=messages,
                    model=settings.euri_model,
                    temperature=0.1,
                    max_tokens=50
                ).strip().lower()
                
                if not state.get("extracted_data"):
                    state["extracted_data"] = {}
                
                state["extracted_data"]["contract_type"] = contract_type
                state["current_step"] = "contract_type_identified"
                
            except Exception as e:
                state["errors"].append(f"Contract type identification failed: {str(e)}")
                logger.error(f"Contract type identification error: {e}")
            
            return state
        
        def extract_parties(state: AgentState) -> AgentState:
            """Extract contract parties."""
            try:
                contract_text = state["contract_text"]
                
                prompt = PromptTemplate(
                    input_variables=["contract_text"],
                    template="""
                    Extract all parties involved in this contract. For each party, identify:
                    - Name
                    - Role (e.g., buyer, seller, contractor, client)
                    - Address (if mentioned)
                    - Contact information (if mentioned)
                    
                    Contract Text:
                    {contract_text}
                    
                    Return the information in JSON format as a list of parties.
                    """
                )
                
                parser = JsonOutputParser()
                
                messages = [
                    {"role": "system", "content": "You are a legal document analysis expert. Extract party information accurately."},
                    {"role": "user", "content": prompt.format(contract_text=contract_text)}
                ]
                
                response = euri_chat_completion(
                    messages=messages,
                    model=settings.euri_model,
                    temperature=0.1,
                    max_tokens=1000
                )
                
                try:
                    parties = parser.parse(response)
                except:
                    # Fallback parsing
                    parties = self._extract_parties_fallback(contract_text)
                
                state["extracted_data"]["parties"] = parties
                state["current_step"] = "parties_extracted"
                
            except Exception as e:
                state["errors"].append(f"Party extraction failed: {str(e)}")
                logger.error(f"Party extraction error: {e}")
            
            return state
        
        def analyze_key_terms(state: AgentState) -> AgentState:
            """Analyze key contract terms."""
            try:
                contract_text = state["contract_text"]
                
                prompt = PromptTemplate(
                    input_variables=["contract_text"],
                    template="""
                    Analyze the key terms and conditions in this contract. Extract:
                    
                    1. Duration and dates (start date, end date, term length)
                    2. Renewal and termination conditions
                    3. Performance obligations for each party
                    4. Deliverables and milestones
                    5. Governing law and jurisdiction
                    
                    Contract Text:
                    {contract_text}
                    
                    Provide a structured analysis in JSON format.
                    """
                )
                
                messages = [
                    {"role": "system", "content": "You are a legal contract analyst. Provide detailed term analysis."},
                    {"role": "user", "content": prompt.format(contract_text=contract_text)}
                ]
                
                response = euri_chat_completion(
                    messages=messages,
                    model=settings.euri_model,
                    temperature=0.2,
                    max_tokens=2000
                )
                
                try:
                    key_terms = JsonOutputParser().parse(response)
                except:
                    key_terms = {"analysis": response}
                
                state["extracted_data"]["key_terms"] = key_terms
                state["current_step"] = "key_terms_analyzed"
                
            except Exception as e:
                state["errors"].append(f"Key terms analysis failed: {str(e)}")
                logger.error(f"Key terms analysis error: {e}")
            
            return state
        
        def analyze_clauses(state: AgentState) -> AgentState:
            """Analyze individual contract clauses."""
            try:
                contract_text = state["contract_text"]
                
                # First, identify and extract clauses
                clauses = self._extract_clauses(contract_text)
                
                analyzed_clauses = []
                
                for clause in clauses:
                    clause_analysis = self._analyze_single_clause(clause)
                    analyzed_clauses.append(clause_analysis)
                
                state["extracted_data"]["clauses"] = analyzed_clauses
                state["current_step"] = "clauses_analyzed"
                
            except Exception as e:
                state["errors"].append(f"Clause analysis failed: {str(e)}")
                logger.error(f"Clause analysis error: {e}")
            
            return state
        
        def assess_risks(state: AgentState) -> AgentState:
            """Assess contract risks."""
            try:
                contract_text = state["contract_text"]
                extracted_data = state.get("extracted_data", {})
                
                prompt = PromptTemplate(
                    input_variables=["contract_text", "contract_type"],
                    template="""
                    Perform a comprehensive risk assessment of this {contract_type} contract.
                    
                    Identify and analyze:
                    1. Legal risks and potential liabilities
                    2. Financial risks and exposure
                    3. Operational risks
                    4. Compliance risks
                    5. Termination risks
                    6. Force majeure and unforeseen circumstances
                    
                    For each risk, provide:
                    - Risk description
                    - Severity level (high/medium/low)
                    - Likelihood (high/medium/low)
                    - Potential impact
                    - Mitigation recommendations
                    
                    Contract Text:
                    {contract_text}
                    
                    Provide analysis in JSON format.
                    """
                )
                
                messages = [
                    {"role": "system", "content": "You are a legal risk assessment expert."},
                    {"role": "user", "content": prompt.format(
                        contract_text=contract_text,
                        contract_type=extracted_data.get("contract_type", "general")
                    )}
                ]
                
                response = euri_chat_completion(
                    messages=messages,
                    model=settings.euri_model,
                    temperature=0.2,
                    max_tokens=2500
                )
                
                try:
                    risks = JsonOutputParser().parse(response)
                except:
                    risks = {"analysis": response}
                
                state["extracted_data"]["risks"] = risks
                state["current_step"] = "risks_assessed"
                
            except Exception as e:
                state["errors"].append(f"Risk assessment failed: {str(e)}")
                logger.error(f"Risk assessment error: {e}")
            
            return state
        
        def extract_financial_terms(state: AgentState) -> AgentState:
            """Extract and analyze financial terms."""
            try:
                contract_text = state["contract_text"]
                
                # Use regex patterns to find financial information
                financial_patterns = {
                    'amounts': r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|EUR|GBP)\b',
                    'percentages': r'\b\d+(?:\.\d+)?%\b',
                    'payment_terms': r'(?:payment|pay|due|invoice|billing).*?(?:\.|;|\n)',
                    'late_fees': r'(?:late fee|penalty|interest).*?(?:\.|;|\n)'
                }
                
                financial_data = {}
                
                for term_type, pattern in financial_patterns.items():
                    matches = re.findall(pattern, contract_text, re.IGNORECASE)
                    financial_data[term_type] = matches
                
                # Use AI to analyze financial terms in context
                prompt = PromptTemplate(
                    input_variables=["contract_text"],
                    template="""
                    Extract and analyze all financial terms from this contract:
                    
                    1. Total contract value
                    2. Payment schedule and terms
                    3. Currency
                    4. Late payment penalties
                    5. Discounts or incentives
                    6. Expense allocations
                    7. Tax responsibilities
                    
                    Contract Text:
                    {contract_text}
                    
                    Provide structured financial analysis in JSON format.
                    """
                )
                
                messages = [
                    {"role": "system", "content": "You are a financial contract analyst."},
                    {"role": "user", "content": prompt.format(contract_text=contract_text)}
                ]
                
                response = euri_chat_completion(
                    messages=messages,
                    model=settings.euri_model,
                    temperature=0.1,
                    max_tokens=1500
                )
                
                try:
                    ai_financial_analysis = JsonOutputParser().parse(response)
                    financial_data.update(ai_financial_analysis)
                except:
                    financial_data["ai_analysis"] = response
                
                state["extracted_data"]["financial_terms"] = financial_data
                state["current_step"] = "financial_terms_extracted"
                
            except Exception as e:
                state["errors"].append(f"Financial terms extraction failed: {str(e)}")
                logger.error(f"Financial terms extraction error: {e}")
            
            return state
        
        def extract_important_dates(state: AgentState) -> AgentState:
            """Extract important dates and deadlines."""
            try:
                contract_text = state["contract_text"]
                
                # Date patterns
                date_patterns = [
                    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
                    r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD or YYYY-MM-DD
                    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                    r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
                ]
                
                dates_found = []
                for pattern in date_patterns:
                    matches = re.findall(pattern, contract_text, re.IGNORECASE)
                    dates_found.extend(matches)
                
                # Use AI to contextualize dates
                prompt = PromptTemplate(
                    input_variables=["contract_text"],
                    template="""
                    Extract all important dates and deadlines from this contract with their context:
                    
                    1. Contract start date
                    2. Contract end date
                    3. Renewal dates
                    4. Payment due dates
                    5. Milestone deadlines
                    6. Notice periods
                    7. Termination dates
                    
                    For each date, provide:
                    - The date
                    - What it represents
                    - Its importance level
                    
                    Contract Text:
                    {contract_text}
                    
                    Provide analysis in JSON format.
                    """
                )
                
                messages = [
                    {"role": "system", "content": "You are a legal date and deadline analyst."},
                    {"role": "user", "content": prompt.format(contract_text=contract_text)}
                ]
                
                response = euri_chat_completion(
                    messages=messages,
                    model=settings.euri_model,
                    temperature=0.1,
                    max_tokens=1500
                )
                
                try:
                    important_dates = JsonOutputParser().parse(response)
                except:
                    important_dates = {"raw_dates": dates_found, "analysis": response}
                
                state["extracted_data"]["important_dates"] = important_dates
                state["current_step"] = "dates_extracted"
                
            except Exception as e:
                state["errors"].append(f"Date extraction failed: {str(e)}")
                logger.error(f"Date extraction error: {e}")
            
            return state
        
        def generate_recommendations(state: AgentState) -> AgentState:
            """Generate recommendations based on analysis."""
            try:
                extracted_data = state.get("extracted_data", {})
                contract_text = state["contract_text"]
                
                prompt = PromptTemplate(
                    input_variables=["contract_analysis", "contract_text"],
                    template="""
                    Based on the contract analysis, provide comprehensive recommendations:
                    
                    Analysis Summary:
                    {contract_analysis}
                    
                    Provide recommendations for:
                    1. Risk mitigation strategies
                    2. Contract improvements
                    3. Negotiation points
                    4. Compliance requirements
                    5. Action items and next steps
                    
                    Format as a list of actionable recommendations.
                    """
                )
                
                messages = [
                    {"role": "system", "content": "You are a senior legal advisor providing contract recommendations."},
                    {"role": "user", "content": prompt.format(
                        contract_analysis=str(extracted_data),
                        contract_text=contract_text[:1000]  # Summary for context
                    )}
                ]
                
                recommendations = euri_chat_completion(
                    messages=messages,
                    model=settings.euri_model,
                    temperature=0.3,
                    max_tokens=2000
                )
                
                state["extracted_data"]["recommendations"] = recommendations
                state["current_step"] = "recommendations_generated"
                
            except Exception as e:
                state["errors"].append(f"Recommendation generation failed: {str(e)}")
                logger.error(f"Recommendation generation error: {e}")
            
            return state
        
        def finalize_analysis(state: AgentState) -> AgentState:
            """Finalize the contract analysis."""
            try:
                extracted_data = state.get("extracted_data", {})
                
                # Create comprehensive analysis result
                analysis_result = ContractAnalysisResult(
                    contract_type=extracted_data.get("contract_type", "unknown"),
                    parties=extracted_data.get("parties", []),
                    key_terms=extracted_data.get("key_terms", {}),
                    clauses=extracted_data.get("clauses", []),
                    obligations=[],  # TODO: Extract from key_terms
                    risks=extracted_data.get("risks", []),
                    important_dates=extracted_data.get("important_dates", []),
                    financial_terms=extracted_data.get("financial_terms", {}),
                    compliance_issues=[],  # TODO: Identify compliance issues
                    recommendations=extracted_data.get("recommendations", ""),
                    summary=self._generate_summary(extracted_data),
                    confidence_score=self._calculate_confidence_score(extracted_data, state.get("errors", []))
                )
                
                state["analysis_results"] = analysis_result
                state["current_step"] = "completed"
                
            except Exception as e:
                state["errors"].append(f"Analysis finalization failed: {str(e)}")
                logger.error(f"Analysis finalization error: {e}")
            
            return state
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("identify_type", identify_contract_type)
        workflow.add_node("extract_parties", extract_parties)
        workflow.add_node("analyze_terms", analyze_key_terms)
        workflow.add_node("analyze_clauses", analyze_clauses)
        workflow.add_node("assess_risks", assess_risks)
        workflow.add_node("extract_financial", extract_financial_terms)
        workflow.add_node("extract_dates", extract_important_dates)
        workflow.add_node("generate_recommendations", generate_recommendations)
        workflow.add_node("finalize", finalize_analysis)
        
        # Add edges
        workflow.set_entry_point("identify_type")
        workflow.add_edge("identify_type", "extract_parties")
        workflow.add_edge("extract_parties", "analyze_terms")
        workflow.add_edge("analyze_terms", "analyze_clauses")
        workflow.add_edge("analyze_clauses", "assess_risks")
        workflow.add_edge("assess_risks", "extract_financial")
        workflow.add_edge("extract_financial", "extract_dates")
        workflow.add_edge("extract_dates", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()

    def _extract_parties_fallback(self, contract_text: str) -> List[Dict[str, Any]]:
        """Fallback method for extracting parties using regex patterns."""
        parties = []

        # Common patterns for party identification
        party_patterns = [
            r'(?:between|by and between)\s+([^,\n]+?)(?:\s+(?:and|&)\s+([^,\n]+?))?(?:\s*,|\s*\()',
            r'(?:party|parties).*?:\s*([^,\n]+)',
            r'(?:client|customer|buyer|seller|contractor|vendor):\s*([^,\n]+)',
            r'(?:company|corporation|llc|inc|ltd).*?([A-Z][^,\n]*(?:company|corporation|llc|inc|ltd)[^,\n]*)'
        ]

        for pattern in party_patterns:
            matches = re.findall(pattern, contract_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    for party_name in match:
                        if party_name.strip():
                            parties.append({
                                "name": party_name.strip(),
                                "role": "party",
                                "address": None,
                                "contact_info": None
                            })
                else:
                    if match.strip():
                        parties.append({
                            "name": match.strip(),
                            "role": "party",
                            "address": None,
                            "contact_info": None
                        })

        # Remove duplicates
        unique_parties = []
        seen_names = set()

        for party in parties:
            name_lower = party["name"].lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                unique_parties.append(party)

        return unique_parties[:10]  # Limit to reasonable number

    def _extract_clauses(self, contract_text: str) -> List[Dict[str, str]]:
        """Extract contract clauses using pattern matching."""
        clauses = []

        # Split text into sections/paragraphs
        sections = re.split(r'\n\s*\n|\n\d+\.|\n[A-Z]+\.', contract_text)

        # Common clause types to look for
        clause_types = {
            'termination': r'(?:termination|terminate|end|expir)',
            'payment': r'(?:payment|pay|fee|cost|price|invoice)',
            'liability': r'(?:liability|liable|responsible|damages)',
            'confidentiality': r'(?:confidential|non-disclosure|proprietary)',
            'intellectual_property': r'(?:intellectual property|copyright|patent|trademark)',
            'force_majeure': r'(?:force majeure|act of god|unforeseeable)',
            'governing_law': r'(?:governing law|jurisdiction|applicable law)',
            'dispute_resolution': r'(?:dispute|arbitration|mediation|litigation)',
            'indemnification': r'(?:indemnif|hold harmless|defend)',
            'warranty': r'(?:warrant|guarantee|represent)'
        }

        for i, section in enumerate(sections):
            if len(section.strip()) < 50:  # Skip very short sections
                continue

            # Identify clause type
            clause_type = "general"
            for type_name, pattern in clause_types.items():
                if re.search(pattern, section, re.IGNORECASE):
                    clause_type = type_name
                    break

            clauses.append({
                "id": f"clause_{i}",
                "type": clause_type,
                "content": section.strip(),
                "section_number": i + 1
            })

        return clauses

    def _analyze_single_clause(self, clause: Dict[str, str]) -> Dict[str, Any]:
        """Analyze a single contract clause."""
        try:
            prompt = PromptTemplate(
                input_variables=["clause_content", "clause_type"],
                template="""
                Analyze this {clause_type} clause from a legal contract:

                Clause Content:
                {clause_content}

                Provide analysis including:
                1. Summary of what this clause means
                2. Importance level (high/medium/low)
                3. Risk level (high/medium/low)
                4. Potential issues or concerns
                5. Recommendations for improvement

                Respond in JSON format.
                """
            )

            messages = [
                {"role": "system", "content": "You are a legal clause analysis expert."},
                {"role": "user", "content": prompt.format(
                    clause_content=clause["content"][:1000],  # Limit length
                    clause_type=clause["type"]
                )}
            ]

            response = euri_chat_completion(
                messages=messages,
                model=settings.euri_model,
                temperature=0.2,
                max_tokens=800
            )

            try:
                analysis = JsonOutputParser().parse(response)
            except:
                analysis = {"analysis": response}

            return {
                "id": clause["id"],
                "type": clause["type"],
                "content": clause["content"],
                "analysis": analysis,
                "section_number": clause.get("section_number")
            }

        except Exception as e:
            logger.error(f"Single clause analysis failed: {e}")
            return {
                "id": clause["id"],
                "type": clause["type"],
                "content": clause["content"],
                "analysis": {"error": str(e)},
                "section_number": clause.get("section_number")
            }

    def _generate_summary(self, extracted_data: Dict[str, Any]) -> str:
        """Generate a comprehensive summary of the contract analysis."""
        try:
            contract_type = extracted_data.get("contract_type", "unknown")
            parties = extracted_data.get("parties", [])
            risks = extracted_data.get("risks", {})
            financial_terms = extracted_data.get("financial_terms", {})

            summary_parts = []

            # Contract type and parties
            party_names = [party.get("name", "Unknown") for party in parties[:3]]
            if len(parties) > 3:
                party_names.append(f"and {len(parties) - 3} others")

            summary_parts.append(
                f"This is a {contract_type} contract between {', '.join(party_names)}."
            )

            # Financial summary
            if financial_terms:
                if "total_value" in financial_terms or "amounts" in financial_terms:
                    summary_parts.append("The contract includes financial obligations and payment terms.")

            # Risk summary
            if isinstance(risks, dict) and risks:
                summary_parts.append("Risk assessment has identified potential areas of concern that require attention.")

            # Key dates
            important_dates = extracted_data.get("important_dates", {})
            if important_dates:
                summary_parts.append("The contract contains important dates and deadlines that must be monitored.")

            return " ".join(summary_parts)

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Contract analysis completed with extracted information available in detailed sections."

    def _calculate_confidence_score(self, extracted_data: Dict[str, Any], errors: List[str]) -> float:
        """Calculate confidence score for the analysis."""
        try:
            score = 1.0

            # Reduce score for errors
            error_penalty = len(errors) * 0.1
            score -= error_penalty

            # Check completeness of extraction
            required_fields = ["contract_type", "parties", "key_terms", "clauses"]
            missing_fields = sum(1 for field in required_fields if not extracted_data.get(field))
            completeness_penalty = missing_fields * 0.15
            score -= completeness_penalty

            # Bonus for successful extractions
            if extracted_data.get("financial_terms"):
                score += 0.05
            if extracted_data.get("important_dates"):
                score += 0.05
            if extracted_data.get("risks"):
                score += 0.05

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Confidence score calculation failed: {e}")
            return 0.5  # Default moderate confidence

    async def analyze(self, contract_text: str) -> ContractAnalysisResult:
        """
        Main analysis method using the LangGraph workflow.

        Args:
            contract_text (str): The contract text to analyze

        Returns:
            ContractAnalysisResult: Comprehensive contract analysis results
        """
        try:
            # Initialize state
            initial_state = AgentState(
                messages=[],
                contract_text=contract_text,
                analysis_results=None,
                current_step="initialized",
                errors=[],
                extracted_data={}
            )

            # Run the analysis workflow
            final_state = await self.graph.ainvoke(initial_state)

            # Return results
            if final_state.get("analysis_results"):
                return final_state["analysis_results"]
            else:
                # Return empty results if analysis failed
                return ContractAnalysisResult(
                    contract_type="unknown",
                    parties=[],
                    key_terms={},
                    clauses=[],
                    obligations=[],
                    risks=[],
                    important_dates=[],
                    financial_terms={},
                    compliance_issues=[],
                    recommendations="Analysis failed. Please review the contract manually.",
                    summary="Contract analysis could not be completed due to errors.",
                    confidence_score=0.0
                )

        except Exception as e:
            logger.error(f"Contract analysis failed: {e}")
            return ContractAnalysisResult(
                contract_type="unknown",
                parties=[],
                key_terms={},
                clauses=[],
                obligations=[],
                risks=[],
                important_dates=[],
                financial_terms={},
                compliance_issues=[],
                recommendations=f"Analysis failed with error: {str(e)}",
                summary="Contract analysis encountered an error and could not be completed.",
                confidence_score=0.0
            )
