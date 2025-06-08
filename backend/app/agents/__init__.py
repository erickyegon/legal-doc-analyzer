"""
AI Agents package for the Legal Intelligence Platform.

This package contains specialized AI agents built with LangChain and LangGraph
for various legal document analysis tasks including multimodal extraction.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

from .multimodal_extraction_agent import MultimodalExtractionAgent
from .contract_analysis_agent import ContractAnalysisAgent
from .risk_assessment_agent import RiskAssessmentAgent
from .compliance_agent import ComplianceAgent
from .agent_orchestrator import AgentOrchestrator

__all__ = [
    "MultimodalExtractionAgent",
    "ContractAnalysisAgent", 
    "RiskAssessmentAgent",
    "ComplianceAgent",
    "AgentOrchestrator"
]
