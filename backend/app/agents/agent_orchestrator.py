"""
Agent Orchestrator for the Legal Intelligence Platform.

This module coordinates multiple specialized agents using LangGraph to provide
comprehensive legal document analysis with multimodal capabilities.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain.schema import BaseMessage

# Agent imports
from .multimodal_extraction_agent import MultimodalExtractionAgent, ExtractionResult
from .contract_analysis_agent import ContractAnalysisAgent, ContractAnalysisResult

# Tool imports
from app.tools.pdf_parser_tool import PDFParserTool
from app.tools.clause_library_tool import ClauseLibraryTool
from app.tools.summarizer_tool import SummarizerTool
from app.tools.ner_tool import NERTool
from app.tools.regex_search_tool import RegexSearchTool
from app.tools.external_api_tool import ExternalAPITool

# Custom imports
from app.models.analysis import AnalysisType
from app.models.document import DocumentType

# Configure logging
logger = logging.getLogger(__name__)


class AnalysisMode(Enum):
    """Analysis modes for different document types."""
    QUICK = "quick"
    COMPREHENSIVE = "comprehensive"
    MULTIMODAL_ONLY = "multimodal_only"
    CONTRACT_FOCUSED = "contract_focused"


@dataclass
class OrchestrationResult:
    """Comprehensive result from agent orchestration."""
    document_path: str
    document_type: DocumentType
    analysis_mode: AnalysisMode
    extraction_results: Optional[ExtractionResult]
    contract_analysis: Optional[ContractAnalysisResult]
    risk_assessment: Optional[Dict[str, Any]]
    compliance_check: Optional[Dict[str, Any]]
    summary: str
    recommendations: List[str]
    confidence_score: float
    processing_time: float
    errors: List[str]


class OrchestratorState(TypedDict):
    """State for the agent orchestrator."""
    messages: List[BaseMessage]
    document_path: str
    document_type: DocumentType
    analysis_type: AnalysisType
    analysis_mode: AnalysisMode
    text_content: str
    extraction_results: Optional[ExtractionResult]
    contract_analysis: Optional[ContractAnalysisResult]
    risk_assessment: Optional[Dict[str, Any]]
    compliance_check: Optional[Dict[str, Any]]
    final_results: Optional[OrchestrationResult]
    current_step: str
    errors: List[str]


class AgentOrchestrator:
    """
    Orchestrates multiple specialized agents for comprehensive legal document analysis.
    
    This orchestrator uses LangGraph to coordinate:
    - Multimodal Extraction Agent (for complex document parsing)
    - Contract Analysis Agent (for contract-specific analysis)
    - Risk Assessment Agent (for legal risk evaluation)
    - Compliance Agent (for regulatory compliance checking)
    """
    
    def __init__(self):
        """Initialize the agent orchestrator."""
        # Initialize agents
        self.multimodal_agent = MultimodalExtractionAgent()
        self.contract_agent = ContractAnalysisAgent()

        # Initialize tools
        self.pdf_parser = PDFParserTool()
        self.clause_library = ClauseLibraryTool()
        self.summarizer = SummarizerTool()
        self.ner_tool = NERTool()
        self.regex_search = RegexSearchTool()
        self.external_api = ExternalAPITool()

        # Build orchestration graph
        self.graph = self._build_orchestration_graph()
        
    def _build_orchestration_graph(self) -> StateGraph:
        """Build the LangGraph workflow for agent orchestration."""
        
        def determine_analysis_strategy(state: OrchestratorState) -> OrchestratorState:
            """Determine the optimal analysis strategy based on document type and requirements."""
            try:
                document_type = state["document_type"]
                analysis_type = state["analysis_type"]
                
                # Determine analysis mode based on document type and analysis type
                if document_type == DocumentType.CONTRACT:
                    if analysis_type == AnalysisType.SUMMARY:
                        state["analysis_mode"] = AnalysisMode.QUICK
                    else:
                        state["analysis_mode"] = AnalysisMode.CONTRACT_FOCUSED
                elif analysis_type == AnalysisType.ENTITY_EXTRACTION:
                    state["analysis_mode"] = AnalysisMode.MULTIMODAL_ONLY
                else:
                    state["analysis_mode"] = AnalysisMode.COMPREHENSIVE
                
                state["current_step"] = "strategy_determined"
                logger.info(f"Analysis strategy: {state['analysis_mode'].value}")
                
            except Exception as e:
                state["errors"].append(f"Strategy determination failed: {str(e)}")
                state["analysis_mode"] = AnalysisMode.QUICK  # Fallback
                logger.error(f"Strategy determination error: {e}")
            
            return state
        
        def extract_multimodal_content(state: OrchestratorState) -> OrchestratorState:
            """Extract multimodal content using the specialized agent."""
            try:
                analysis_mode = state["analysis_mode"]
                
                # Skip multimodal extraction for quick analysis unless specifically needed
                if (analysis_mode == AnalysisMode.QUICK and 
                    state["analysis_type"] not in [AnalysisType.ENTITY_EXTRACTION, AnalysisType.DATE_EXTRACTION]):
                    state["current_step"] = "multimodal_skipped"
                    return state
                
                document_path = state["document_path"]
                
                # Run multimodal extraction
                extraction_results = self.multimodal_agent.extract(document_path)
                state["extraction_results"] = extraction_results
                
                # Update text content with extracted text
                if extraction_results and extraction_results.text_content:
                    state["text_content"] = extraction_results.text_content
                
                state["current_step"] = "multimodal_extracted"
                logger.info("Multimodal extraction completed")
                
            except Exception as e:
                state["errors"].append(f"Multimodal extraction failed: {str(e)}")
                logger.error(f"Multimodal extraction error: {e}")
            
            return state
        
        def analyze_contract_content(state: OrchestratorState) -> OrchestratorState:
            """Analyze contract content using the contract analysis agent."""
            try:
                analysis_mode = state["analysis_mode"]
                document_type = state["document_type"]
                
                # Only run contract analysis for contracts or comprehensive analysis
                if (document_type != DocumentType.CONTRACT and 
                    analysis_mode not in [AnalysisMode.COMPREHENSIVE, AnalysisMode.CONTRACT_FOCUSED]):
                    state["current_step"] = "contract_analysis_skipped"
                    return state
                
                text_content = state["text_content"]
                if not text_content:
                    state["errors"].append("No text content available for contract analysis")
                    return state
                
                # Run contract analysis
                contract_analysis = self.contract_agent.analyze(text_content)
                state["contract_analysis"] = contract_analysis
                
                state["current_step"] = "contract_analyzed"
                logger.info("Contract analysis completed")
                
            except Exception as e:
                state["errors"].append(f"Contract analysis failed: {str(e)}")
                logger.error(f"Contract analysis error: {e}")
            
            return state
        
        def assess_legal_risks(state: OrchestratorState) -> OrchestratorState:
            """Assess legal risks in the document."""
            try:
                analysis_mode = state["analysis_mode"]
                analysis_type = state["analysis_type"]
                
                # Skip risk assessment for quick mode unless specifically requested
                if (analysis_mode == AnalysisMode.QUICK and 
                    analysis_type != AnalysisType.RISK_ASSESSMENT):
                    state["current_step"] = "risk_assessment_skipped"
                    return state
                
                # Use contract analysis risks if available, otherwise perform basic risk assessment
                if state.get("contract_analysis") and state["contract_analysis"].risks:
                    risk_assessment = {
                        "source": "contract_analysis",
                        "risks": state["contract_analysis"].risks,
                        "confidence": state["contract_analysis"].confidence_score
                    }
                else:
                    # Perform basic risk assessment
                    risk_assessment = self._perform_basic_risk_assessment(
                        state["text_content"], 
                        state["document_type"]
                    )
                
                state["risk_assessment"] = risk_assessment
                state["current_step"] = "risks_assessed"
                logger.info("Risk assessment completed")
                
            except Exception as e:
                state["errors"].append(f"Risk assessment failed: {str(e)}")
                logger.error(f"Risk assessment error: {e}")
            
            return state
        
        def check_compliance(state: OrchestratorState) -> OrchestratorState:
            """Check regulatory compliance."""
            try:
                analysis_mode = state["analysis_mode"]
                analysis_type = state["analysis_type"]
                
                # Skip compliance check for quick mode unless specifically requested
                if (analysis_mode == AnalysisMode.QUICK and 
                    analysis_type != AnalysisType.COMPLIANCE_CHECK):
                    state["current_step"] = "compliance_skipped"
                    return state
                
                # Perform compliance check
                compliance_check = self._perform_compliance_check(
                    state["text_content"],
                    state["document_type"],
                    state.get("contract_analysis")
                )
                
                state["compliance_check"] = compliance_check
                state["current_step"] = "compliance_checked"
                logger.info("Compliance check completed")
                
            except Exception as e:
                state["errors"].append(f"Compliance check failed: {str(e)}")
                logger.error(f"Compliance check error: {e}")
            
            return state
        
        def synthesize_results(state: OrchestratorState) -> OrchestratorState:
            """Synthesize all analysis results into a comprehensive report."""
            try:
                # Gather all results
                extraction_results = state.get("extraction_results")
                contract_analysis = state.get("contract_analysis")
                risk_assessment = state.get("risk_assessment")
                compliance_check = state.get("compliance_check")
                
                # Generate comprehensive summary
                summary = self._generate_comprehensive_summary(
                    extraction_results, contract_analysis, risk_assessment, compliance_check
                )
                
                # Generate recommendations
                recommendations = self._generate_comprehensive_recommendations(
                    extraction_results, contract_analysis, risk_assessment, compliance_check
                )
                
                # Calculate overall confidence score
                confidence_score = self._calculate_overall_confidence(
                    extraction_results, contract_analysis, risk_assessment, compliance_check, state["errors"]
                )
                
                # Create final results
                final_results = OrchestrationResult(
                    document_path=state["document_path"],
                    document_type=state["document_type"],
                    analysis_mode=state["analysis_mode"],
                    extraction_results=extraction_results,
                    contract_analysis=contract_analysis,
                    risk_assessment=risk_assessment,
                    compliance_check=compliance_check,
                    summary=summary,
                    recommendations=recommendations,
                    confidence_score=confidence_score,
                    processing_time=0.0,  # Will be calculated by caller
                    errors=state["errors"]
                )
                
                state["final_results"] = final_results
                state["current_step"] = "completed"
                logger.info("Results synthesis completed")
                
            except Exception as e:
                state["errors"].append(f"Results synthesis failed: {str(e)}")
                logger.error(f"Results synthesis error: {e}")
            
            return state
        
        # Build the graph
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes
        workflow.add_node("determine_strategy", determine_analysis_strategy)
        workflow.add_node("extract_multimodal", extract_multimodal_content)
        workflow.add_node("analyze_contract", analyze_contract_content)
        workflow.add_node("assess_risks", assess_legal_risks)
        workflow.add_node("check_compliance", check_compliance)
        workflow.add_node("synthesize", synthesize_results)
        
        # Add edges
        workflow.set_entry_point("determine_strategy")
        workflow.add_edge("determine_strategy", "extract_multimodal")
        workflow.add_edge("extract_multimodal", "analyze_contract")
        workflow.add_edge("analyze_contract", "assess_risks")
        workflow.add_edge("assess_risks", "check_compliance")
        workflow.add_edge("check_compliance", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def _perform_basic_risk_assessment(self, text_content: str, document_type: DocumentType) -> Dict[str, Any]:
        """Perform basic risk assessment when contract analysis is not available."""
        # This is a simplified implementation
        # In a full implementation, this would use another specialized agent
        
        risks = []
        
        # Basic keyword-based risk detection
        risk_keywords = {
            "high": ["liability", "penalty", "damages", "breach", "default", "termination"],
            "medium": ["obligation", "requirement", "must", "shall", "responsible"],
            "low": ["may", "should", "recommend", "suggest"]
        }
        
        text_lower = text_content.lower()
        
        for risk_level, keywords in risk_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    risks.append({
                        "type": "keyword_detected",
                        "keyword": keyword,
                        "risk_level": risk_level,
                        "description": f"Document contains '{keyword}' which may indicate {risk_level} risk"
                    })
        
        return {
            "source": "basic_assessment",
            "risks": risks,
            "confidence": 0.3  # Low confidence for basic assessment
        }
    
    def _perform_compliance_check(self, text_content: str, document_type: DocumentType, 
                                 contract_analysis: Optional[ContractAnalysisResult]) -> Dict[str, Any]:
        """Perform basic compliance check."""
        # This is a simplified implementation
        # In a full implementation, this would use a specialized compliance agent
        
        compliance_issues = []
        
        # Basic compliance checks based on document type
        if document_type == DocumentType.CONTRACT:
            # Check for required contract elements
            required_elements = ["parties", "consideration", "terms", "signatures"]
            text_lower = text_content.lower()
            
            for element in required_elements:
                if element not in text_lower:
                    compliance_issues.append({
                        "type": "missing_element",
                        "element": element,
                        "severity": "medium",
                        "description": f"Contract may be missing required element: {element}"
                    })
        
        return {
            "source": "basic_compliance",
            "issues": compliance_issues,
            "confidence": 0.4
        }

    def _generate_comprehensive_summary(self, extraction_results: Optional[ExtractionResult],
                                      contract_analysis: Optional[ContractAnalysisResult],
                                      risk_assessment: Optional[Dict[str, Any]],
                                      compliance_check: Optional[Dict[str, Any]]) -> str:
        """Generate a comprehensive summary of all analysis results."""
        summary_parts = []

        # Document structure summary
        if extraction_results:
            if extraction_results.tables:
                summary_parts.append(f"Document contains {len(extraction_results.tables)} tables with structured data.")
            if extraction_results.images:
                summary_parts.append(f"Document includes {len(extraction_results.images)} images.")
            if extraction_results.signatures:
                summary_parts.append(f"Document has {len(extraction_results.signatures)} detected signatures.")

        # Contract analysis summary
        if contract_analysis:
            summary_parts.append(f"Contract analysis identified this as a {contract_analysis.contract_type} contract.")
            if contract_analysis.parties:
                party_count = len(contract_analysis.parties)
                summary_parts.append(f"Contract involves {party_count} parties.")
            if contract_analysis.financial_terms:
                summary_parts.append("Contract includes financial terms and payment obligations.")

        # Risk summary
        if risk_assessment and risk_assessment.get("risks"):
            risk_count = len(risk_assessment["risks"])
            summary_parts.append(f"Risk assessment identified {risk_count} potential risk areas.")

        # Compliance summary
        if compliance_check and compliance_check.get("issues"):
            issue_count = len(compliance_check["issues"])
            summary_parts.append(f"Compliance review found {issue_count} areas requiring attention.")

        if not summary_parts:
            return "Document analysis completed. Detailed results are available in individual sections."

        return " ".join(summary_parts)

    def _generate_comprehensive_recommendations(self, extraction_results: Optional[ExtractionResult],
                                              contract_analysis: Optional[ContractAnalysisResult],
                                              risk_assessment: Optional[Dict[str, Any]],
                                              compliance_check: Optional[Dict[str, Any]]) -> List[str]:
        """Generate comprehensive recommendations based on all analysis results."""
        recommendations = []

        # Multimodal extraction recommendations
        if extraction_results:
            if extraction_results.tables:
                recommendations.append("Review extracted tables for accuracy and completeness of financial data.")
            if extraction_results.signatures:
                recommendations.append("Verify the authenticity and validity of detected signatures.")
            if extraction_results.errors:
                recommendations.append("Some content extraction issues were encountered - manual review recommended.")

        # Contract analysis recommendations
        if contract_analysis:
            if isinstance(contract_analysis.recommendations, str):
                recommendations.append(contract_analysis.recommendations)
            elif isinstance(contract_analysis.recommendations, list):
                recommendations.extend(contract_analysis.recommendations)

            # Risk-based recommendations
            if contract_analysis.risks:
                high_risks = [r for r in contract_analysis.risks if isinstance(r, dict) and r.get("severity") == "high"]
                if high_risks:
                    recommendations.append("Address high-risk clauses identified in the contract analysis.")

        # Risk assessment recommendations
        if risk_assessment and risk_assessment.get("risks"):
            high_risks = [r for r in risk_assessment["risks"] if r.get("risk_level") == "high"]
            if high_risks:
                recommendations.append("Implement risk mitigation strategies for identified high-risk areas.")

        # Compliance recommendations
        if compliance_check and compliance_check.get("issues"):
            critical_issues = [i for i in compliance_check["issues"] if i.get("severity") == "high"]
            if critical_issues:
                recommendations.append("Address critical compliance issues before proceeding with the contract.")

        # General recommendations
        if not recommendations:
            recommendations.append("Document appears to be in good order. Regular review is recommended.")

        return recommendations

    def _calculate_overall_confidence(self, extraction_results: Optional[ExtractionResult],
                                    contract_analysis: Optional[ContractAnalysisResult],
                                    risk_assessment: Optional[Dict[str, Any]],
                                    compliance_check: Optional[Dict[str, Any]],
                                    errors: List[str]) -> float:
        """Calculate overall confidence score for the analysis."""
        confidence_scores = []

        # Extraction confidence
        if extraction_results and hasattr(extraction_results, 'metadata'):
            # Use extraction quality metrics
            metadata = extraction_results.metadata
            if metadata.get('quality_metrics'):
                quality = metadata['quality_metrics']
                extraction_confidence = 0.8  # Base confidence
                if quality.get('has_tables'):
                    extraction_confidence += 0.1
                if quality.get('has_signatures'):
                    extraction_confidence += 0.1
                confidence_scores.append(min(1.0, extraction_confidence))

        # Contract analysis confidence
        if contract_analysis:
            confidence_scores.append(contract_analysis.confidence_score)

        # Risk assessment confidence
        if risk_assessment and risk_assessment.get("confidence"):
            confidence_scores.append(risk_assessment["confidence"])

        # Compliance check confidence
        if compliance_check and compliance_check.get("confidence"):
            confidence_scores.append(compliance_check["confidence"])

        # Calculate weighted average
        if confidence_scores:
            overall_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            overall_confidence = 0.5  # Default moderate confidence

        # Reduce confidence based on errors
        error_penalty = min(len(errors) * 0.1, 0.3)  # Max 30% penalty
        overall_confidence = max(0.0, overall_confidence - error_penalty)

        return round(overall_confidence, 2)

    async def orchestrate_analysis(self, document_path: str, document_type: DocumentType,
                                 analysis_type: AnalysisType, text_content: str = "") -> OrchestrationResult:
        """
        Main orchestration method that coordinates all agents.

        Args:
            document_path (str): Path to the document
            document_type (DocumentType): Type of document
            analysis_type (AnalysisType): Type of analysis requested
            text_content (str): Pre-extracted text content (optional)

        Returns:
            OrchestrationResult: Comprehensive analysis results
        """
        import time
        start_time = time.time()

        try:
            # Initialize state
            initial_state = OrchestratorState(
                messages=[],
                document_path=document_path,
                document_type=document_type,
                analysis_type=analysis_type,
                analysis_mode=AnalysisMode.COMPREHENSIVE,  # Will be determined
                text_content=text_content,
                extraction_results=None,
                contract_analysis=None,
                risk_assessment=None,
                compliance_check=None,
                final_results=None,
                current_step="initialized",
                errors=[]
            )

            # Run the orchestration workflow
            final_state = await self.graph.ainvoke(initial_state)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Return results
            if final_state.get("final_results"):
                final_state["final_results"].processing_time = processing_time
                return final_state["final_results"]
            else:
                # Return error results if orchestration failed
                return OrchestrationResult(
                    document_path=document_path,
                    document_type=document_type,
                    analysis_mode=AnalysisMode.QUICK,
                    extraction_results=None,
                    contract_analysis=None,
                    risk_assessment=None,
                    compliance_check=None,
                    summary="Analysis orchestration failed due to errors.",
                    recommendations=["Manual review of the document is recommended."],
                    confidence_score=0.0,
                    processing_time=processing_time,
                    errors=final_state.get("errors", ["Unknown orchestration error"])
                )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Analysis orchestration failed: {e}")

            return OrchestrationResult(
                document_path=document_path,
                document_type=document_type,
                analysis_mode=AnalysisMode.QUICK,
                extraction_results=None,
                contract_analysis=None,
                risk_assessment=None,
                compliance_check=None,
                summary=f"Analysis orchestration encountered an error: {str(e)}",
                recommendations=["Manual review of the document is recommended."],
                confidence_score=0.0,
                processing_time=processing_time,
                errors=[str(e)]
            )

    def get_analysis_capabilities(self) -> Dict[str, Any]:
        """Get information about the orchestrator's analysis capabilities."""
        return {
            "supported_document_types": [dt.value for dt in DocumentType],
            "supported_analysis_types": [at.value for at in AnalysisType],
            "analysis_modes": [am.value for am in AnalysisMode],
            "agents": {
                "multimodal_extraction": {
                    "capabilities": [
                        "Text extraction with layout preservation",
                        "Table extraction and parsing",
                        "Image and diagram extraction",
                        "Signature detection",
                        "Document layout analysis"
                    ]
                },
                "contract_analysis": {
                    "capabilities": [
                        "Contract type identification",
                        "Party extraction",
                        "Clause analysis",
                        "Risk assessment",
                        "Financial terms extraction",
                        "Important dates identification"
                    ]
                },
                "risk_assessment": {
                    "capabilities": [
                        "Legal risk identification",
                        "Risk severity assessment",
                        "Mitigation recommendations"
                    ]
                },
                "compliance_check": {
                    "capabilities": [
                        "Regulatory compliance verification",
                        "Required element checking",
                        "Compliance issue identification"
                    ]
                }
            },
            "features": {
                "multimodal_processing": True,
                "table_extraction": True,
                "signature_detection": True,
                "layout_analysis": True,
                "contract_specialization": True,
                "risk_assessment": True,
                "compliance_checking": True,
                "comprehensive_reporting": True
            }
        }
