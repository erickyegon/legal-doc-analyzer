"""
LangServe integration for the Legal Intelligence Platform.

This module provides LangServe endpoints for the AI agents, allowing them
to be served as REST APIs with automatic OpenAPI documentation.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# Agent imports
from app.agents.multimodal_extraction_agent import MultimodalExtractionAgent
from app.agents.contract_analysis_agent import ContractAnalysisAgent
from app.agents.agent_orchestrator import AgentOrchestrator

# Configure logging
logger = logging.getLogger(__name__)


# Pydantic models for LangServe endpoints
class DocumentAnalysisRequest(BaseModel):
    """Request model for document analysis."""
    document_path: str = Field(description="Path to the document file")
    analysis_type: str = Field(description="Type of analysis to perform", default="comprehensive")


class MultimodalExtractionRequest(BaseModel):
    """Request model for multimodal extraction."""
    document_path: str = Field(description="Path to the document file")


class ContractAnalysisRequest(BaseModel):
    """Request model for contract analysis."""
    contract_text: str = Field(description="Contract text to analyze")


class ExtractionResponse(BaseModel):
    """Response model for extraction results."""
    text_content: str
    tables: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    signatures: List[Dict[str, Any]]
    diagrams: List[Dict[str, Any]]
    layout_elements: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class ContractAnalysisResponse(BaseModel):
    """Response model for contract analysis."""
    contract_type: str
    parties: List[Dict[str, Any]]
    key_terms: Dict[str, Any]
    clauses: List[Dict[str, Any]]
    risks: List[Dict[str, Any]]
    important_dates: List[Dict[str, Any]]
    financial_terms: Dict[str, Any]
    recommendations: str
    summary: str
    confidence_score: float


class LangServeAgentWrapper:
    """Wrapper class to make agents compatible with LangServe."""
    
    def __init__(self):
        """Initialize the agent wrapper."""
        self.multimodal_agent = MultimodalExtractionAgent()
        self.contract_agent = ContractAnalysisAgent()
        self.orchestrator = AgentOrchestrator()
    
    async def extract_multimodal(self, request: MultimodalExtractionRequest) -> Dict[str, Any]:
        """
        Wrapper for multimodal extraction agent.
        
        Args:
            request: Multimodal extraction request
            
        Returns:
            Dict: Extraction results
        """
        try:
            result = await self.multimodal_agent.extract(request.document_path)
            
            return {
                "text_content": result.text_content,
                "tables": result.tables,
                "images": result.images,
                "signatures": result.signatures,
                "diagrams": result.diagrams,
                "layout_elements": result.layout_elements,
                "metadata": result.metadata
            }
            
        except Exception as e:
            logger.error(f"Multimodal extraction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def analyze_contract(self, request: ContractAnalysisRequest) -> Dict[str, Any]:
        """
        Wrapper for contract analysis agent.
        
        Args:
            request: Contract analysis request
            
        Returns:
            Dict: Contract analysis results
        """
        try:
            result = await self.contract_agent.analyze(request.contract_text)
            
            return {
                "contract_type": result.contract_type,
                "parties": result.parties,
                "key_terms": result.key_terms,
                "clauses": result.clauses,
                "risks": result.risks,
                "important_dates": result.important_dates,
                "financial_terms": result.financial_terms,
                "recommendations": result.recommendations,
                "summary": result.summary,
                "confidence_score": result.confidence_score
            }
            
        except Exception as e:
            logger.error(f"Contract analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def orchestrate_analysis(self, request: DocumentAnalysisRequest) -> Dict[str, Any]:
        """
        Wrapper for agent orchestrator.
        
        Args:
            request: Document analysis request
            
        Returns:
            Dict: Orchestrated analysis results
        """
        try:
            from app.models.document import DocumentType
            from app.models.analysis import AnalysisType
            
            # Map string values to enums
            document_type = DocumentType.OTHER  # Default
            analysis_type = AnalysisType.SUMMARY  # Default
            
            # Try to map analysis type
            analysis_type_mapping = {
                "summary": AnalysisType.SUMMARY,
                "risk_assessment": AnalysisType.RISK_ASSESSMENT,
                "clause_analysis": AnalysisType.CLAUSE_ANALYSIS,
                "compliance_check": AnalysisType.COMPLIANCE_CHECK,
                "entity_extraction": AnalysisType.ENTITY_EXTRACTION,
                "date_extraction": AnalysisType.DATE_EXTRACTION,
                "comprehensive": AnalysisType.SUMMARY  # Default for comprehensive
            }
            
            if request.analysis_type in analysis_type_mapping:
                analysis_type = analysis_type_mapping[request.analysis_type]
            
            result = await self.orchestrator.orchestrate_analysis(
                document_path=request.document_path,
                document_type=document_type,
                analysis_type=analysis_type
            )
            
            return {
                "document_path": result.document_path,
                "document_type": result.document_type.value,
                "analysis_mode": result.analysis_mode.value,
                "extraction_results": result.extraction_results.__dict__ if result.extraction_results else None,
                "contract_analysis": result.contract_analysis.__dict__ if result.contract_analysis else None,
                "risk_assessment": result.risk_assessment,
                "compliance_check": result.compliance_check,
                "summary": result.summary,
                "recommendations": result.recommendations,
                "confidence_score": result.confidence_score,
                "processing_time": result.processing_time,
                "errors": result.errors
            }
            
        except Exception as e:
            logger.error(f"Orchestrated analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


def create_langserve_app() -> FastAPI:
    """
    Create FastAPI app with LangServe routes for AI agents.
    
    Returns:
        FastAPI: Configured FastAPI app with LangServe routes
    """
    app = FastAPI(
        title="Legal Intelligence Platform - AI Agents API",
        description="LangServe API for Legal Intelligence Platform AI Agents",
        version="1.0.0"
    )
    
    # Initialize agent wrapper
    agent_wrapper = LangServeAgentWrapper()
    
    # Create runnables for each agent
    multimodal_runnable = RunnableLambda(agent_wrapper.extract_multimodal)
    contract_runnable = RunnableLambda(agent_wrapper.analyze_contract)
    orchestrator_runnable = RunnableLambda(agent_wrapper.orchestrate_analysis)
    
    # Add LangServe routes
    add_routes(
        app,
        multimodal_runnable,
        path="/multimodal-extraction",
        input_type=MultimodalExtractionRequest,
        output_type=Dict[str, Any],
        config_keys=["metadata", "tags"]
    )
    
    add_routes(
        app,
        contract_runnable,
        path="/contract-analysis",
        input_type=ContractAnalysisRequest,
        output_type=Dict[str, Any],
        config_keys=["metadata", "tags"]
    )
    
    add_routes(
        app,
        orchestrator_runnable,
        path="/orchestrated-analysis",
        input_type=DocumentAnalysisRequest,
        output_type=Dict[str, Any],
        config_keys=["metadata", "tags"]
    )
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint for LangServe app."""
        return {
            "status": "healthy",
            "service": "Legal Intelligence Platform - AI Agents",
            "version": "1.0.0",
            "available_endpoints": [
                "/multimodal-extraction",
                "/contract-analysis", 
                "/orchestrated-analysis"
            ]
        }
    
    # Add capabilities endpoint
    @app.get("/capabilities")
    async def get_capabilities():
        """Get information about agent capabilities."""
        return agent_wrapper.orchestrator.get_analysis_capabilities()
    
    return app


# Create the LangServe app instance
langserve_app = create_langserve_app()


if __name__ == "__main__":
    import uvicorn
    
    # Run the LangServe app
    uvicorn.run(
        "app.langserve_app:langserve_app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
