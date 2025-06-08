"""
Analysis service for the Legal Intelligence Platform.

This module handles the processing of document analysis requests,
coordinating between AI agents and document content to generate insights.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import logging
import time
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.analysis import Analysis, AnalysisStatus, AnalysisType
from app.models.agent import Agent
from app.models.document import Document, DocumentType
from app.euri_client import euri_chat_completion
from app.services.document_processor import DocumentProcessor
from app.agents.agent_orchestrator import AgentOrchestrator

# Configure logging
logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Service class for handling document analysis operations.
    
    This service coordinates the analysis process between documents,
    AI agents, and the EURI AI client to generate comprehensive
    legal document insights.
    """
    
    def __init__(self):
        """Initialize the analysis service."""
        self.document_processor = DocumentProcessor()
        self.agent_orchestrator = AgentOrchestrator()
    
    async def process_analysis(self, analysis_id: int, agent_id: int, document_path: str):
        """
        Process a document analysis request.
        
        This method runs as a background task to perform the actual
        analysis using the specified agent and document.
        
        Args:
            analysis_id (int): ID of the analysis to process
            agent_id (int): ID of the agent to use for analysis
            document_path (str): Path to the document file
        """
        db = SessionLocal()
        
        try:
            # Get analysis and document from database
            analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
            if not analysis:
                logger.error(f"Analysis {analysis_id} not found")
                return

            document = db.query(Document).filter(Document.id == analysis.document_id).first()
            if not document:
                logger.error(f"Document for analysis {analysis_id} not found")
                return

            # Mark analysis as started
            analysis.mark_started()
            db.commit()

            # Use the agent orchestrator for comprehensive analysis
            orchestration_result = await self.agent_orchestrator.orchestrate_analysis(
                document_path=document_path,
                document_type=document.document_type,
                analysis_type=analysis.analysis_type,
                text_content=""  # Let orchestrator extract content
            )

            if orchestration_result and orchestration_result.summary:
                # Convert orchestration result to analysis content
                analysis_content = self._format_orchestration_result(orchestration_result)

                # Mark analysis as completed
                analysis.mark_completed(
                    content=analysis_content,
                    summary=orchestration_result.summary,
                    confidence_score=orchestration_result.confidence_score
                )

                # Update processing time
                analysis.processing_time = orchestration_result.processing_time

                # Update agent usage statistics if agent was used
                if agent_id:
                    agent = db.query(Agent).filter(Agent.id == agent_id).first()
                    if agent:
                        agent.update_usage_stats(orchestration_result.processing_time, True)

                logger.info(f"Analysis {analysis_id} completed successfully using orchestrator")
            else:
                error_msg = "Orchestrated analysis failed"
                if orchestration_result and orchestration_result.errors:
                    error_msg += f": {'; '.join(orchestration_result.errors)}"

                analysis.mark_failed(error_msg)

                # Update agent usage statistics
                if agent_id:
                    agent = db.query(Agent).filter(Agent.id == agent_id).first()
                    if agent:
                        agent.update_usage_stats(0, False)

                logger.error(f"Analysis {analysis_id} failed: {error_msg}")

            db.commit()
            
        except Exception as e:
            logger.error(f"Error processing analysis {analysis_id}: {e}")
            
            # Mark analysis as failed
            if analysis:
                analysis.mark_failed(f"Processing error: {str(e)}")
                db.commit()
        
        finally:
            db.close()

    def _format_orchestration_result(self, result) -> str:
        """
        Format orchestration result into analysis content.

        Args:
            result: OrchestrationResult from agent orchestrator

        Returns:
            str: Formatted analysis content
        """
        content_parts = []

        # Add summary
        content_parts.append(f"## Analysis Summary\n{result.summary}\n")

        # Add multimodal extraction results
        if result.extraction_results:
            content_parts.append("## Document Structure Analysis")

            if result.extraction_results.tables:
                content_parts.append(f"### Tables ({len(result.extraction_results.tables)} found)")
                for i, table in enumerate(result.extraction_results.tables[:3]):  # Limit to first 3
                    content_parts.append(f"**Table {i+1}** ({table.get('method', 'unknown')} extraction)")
                    content_parts.append(f"- Shape: {table.get('shape', 'unknown')}")
                    content_parts.append(f"- Page: {table.get('page', 'unknown')}")
                    if table.get('accuracy'):
                        content_parts.append(f"- Accuracy: {table['accuracy']:.2f}")

            if result.extraction_results.signatures:
                content_parts.append(f"### Signatures ({len(result.extraction_results.signatures)} detected)")
                for i, sig in enumerate(result.extraction_results.signatures[:3]):
                    content_parts.append(f"**Signature {i+1}**")
                    content_parts.append(f"- Page: {sig.get('page', 'unknown')}")
                    content_parts.append(f"- Confidence: {sig.get('confidence', 0):.2f}")
                    if sig.get('extracted_text'):
                        content_parts.append(f"- Text: {sig['extracted_text']}")

            if result.extraction_results.images:
                content_parts.append(f"### Images ({len(result.extraction_results.images)} found)")

            if result.extraction_results.diagrams:
                content_parts.append(f"### Diagrams ({len(result.extraction_results.diagrams)} found)")

        # Add contract analysis results
        if result.contract_analysis:
            content_parts.append("## Contract Analysis")
            content_parts.append(f"**Contract Type:** {result.contract_analysis.contract_type}")

            if result.contract_analysis.parties:
                content_parts.append(f"### Parties ({len(result.contract_analysis.parties)})")
                for party in result.contract_analysis.parties[:5]:  # Limit to first 5
                    content_parts.append(f"- **{party.get('name', 'Unknown')}** ({party.get('role', 'Unknown role')})")

            if result.contract_analysis.key_terms:
                content_parts.append("### Key Terms")
                terms = result.contract_analysis.key_terms
                if isinstance(terms, dict):
                    for key, value in list(terms.items())[:5]:  # Limit to first 5
                        content_parts.append(f"- **{key}:** {value}")

            if result.contract_analysis.financial_terms:
                content_parts.append("### Financial Terms")
                financial = result.contract_analysis.financial_terms
                if isinstance(financial, dict):
                    for key, value in list(financial.items())[:5]:
                        content_parts.append(f"- **{key}:** {value}")

        # Add risk assessment
        if result.risk_assessment:
            content_parts.append("## Risk Assessment")
            risks = result.risk_assessment.get('risks', [])
            if risks:
                for risk in risks[:5]:  # Limit to first 5 risks
                    if isinstance(risk, dict):
                        risk_level = risk.get('risk_level', risk.get('severity', 'unknown'))
                        content_parts.append(f"- **{risk_level.upper()} RISK:** {risk.get('description', 'No description')}")

        # Add compliance check
        if result.compliance_check:
            content_parts.append("## Compliance Review")
            issues = result.compliance_check.get('issues', [])
            if issues:
                for issue in issues[:5]:  # Limit to first 5 issues
                    if isinstance(issue, dict):
                        severity = issue.get('severity', 'unknown')
                        content_parts.append(f"- **{severity.upper()}:** {issue.get('description', 'No description')}")
            else:
                content_parts.append("No significant compliance issues identified.")

        # Add recommendations
        if result.recommendations:
            content_parts.append("## Recommendations")
            for i, rec in enumerate(result.recommendations[:10], 1):  # Limit to first 10
                content_parts.append(f"{i}. {rec}")

        # Add metadata
        content_parts.append("## Analysis Metadata")
        content_parts.append(f"- **Analysis Mode:** {result.analysis_mode.value}")
        content_parts.append(f"- **Processing Time:** {result.processing_time:.2f} seconds")
        content_parts.append(f"- **Confidence Score:** {result.confidence_score:.2f}")

        if result.errors:
            content_parts.append("### Errors Encountered")
            for error in result.errors:
                content_parts.append(f"- {error}")

        return "\n\n".join(content_parts)
    
    async def _extract_document_content(self, document_path: str) -> Optional[str]:
        """
        Extract text content from document file.
        
        Args:
            document_path (str): Path to the document file
            
        Returns:
            str: Extracted text content, or None if extraction fails
        """
        try:
            return await self.document_processor.extract_text(document_path)
        except Exception as e:
            logger.error(f"Failed to extract content from {document_path}: {e}")
            return None
    
    async def _generate_analysis(
        self, 
        agent: Agent, 
        analysis_type: str, 
        document_content: str
    ) -> Optional[str]:
        """
        Generate analysis using the specified agent and document content.
        
        Args:
            agent (Agent): AI agent to use for analysis
            analysis_type (str): Type of analysis to perform
            document_content (str): Document text content
            
        Returns:
            str: Generated analysis content, or None if generation fails
        """
        try:
            # Get prompt template for the agent
            prompt_template = agent.get_prompt_template()
            
            # Format prompt with document content
            prompt = prompt_template.format(document_content=document_content)
            
            # Prepare messages for EURI API
            messages = [
                {
                    "role": "system",
                    "content": f"You are a specialized legal AI assistant performing {analysis_type} analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Call EURI API
            start_time = time.time()
            
            analysis_result = euri_chat_completion(
                messages=messages,
                model=agent.model_name,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"Analysis generated in {processing_time:.2f} seconds")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to generate analysis: {e}")
            return None
    
    def _extract_summary(self, analysis_content: str) -> str:
        """
        Extract a summary from the analysis content.
        
        Args:
            analysis_content (str): Full analysis content
            
        Returns:
            str: Summary of the analysis (first 500 characters)
        """
        if not analysis_content:
            return ""
        
        # Simple summary extraction - take first paragraph or 500 characters
        lines = analysis_content.split('\n')
        summary_lines = []
        char_count = 0
        
        for line in lines:
            if char_count + len(line) > 500:
                break
            summary_lines.append(line)
            char_count += len(line)
        
        summary = '\n'.join(summary_lines)
        
        # If summary is too short, take first 500 characters
        if len(summary) < 100:
            summary = analysis_content[:500]
        
        return summary.strip()
    
    def _calculate_confidence_score(self, analysis_content: str) -> float:
        """
        Calculate a confidence score for the analysis.
        
        This is a simplified implementation. In a real system,
        you might use more sophisticated methods to assess confidence.
        
        Args:
            analysis_content (str): Analysis content
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        if not analysis_content:
            return 0.0
        
        # Simple heuristic based on content length and structure
        content_length = len(analysis_content)
        
        # Base confidence on content length
        if content_length < 100:
            base_confidence = 0.3
        elif content_length < 500:
            base_confidence = 0.6
        elif content_length < 1000:
            base_confidence = 0.8
        else:
            base_confidence = 0.9
        
        # Adjust based on content structure
        lines = analysis_content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if len(non_empty_lines) > 5:
            structure_bonus = 0.1
        else:
            structure_bonus = 0.0
        
        # Check for key legal terms (simple implementation)
        legal_terms = [
            'contract', 'agreement', 'clause', 'liability', 'compliance',
            'regulation', 'statute', 'precedent', 'jurisdiction', 'damages'
        ]
        
        term_count = sum(1 for term in legal_terms if term.lower() in analysis_content.lower())
        term_bonus = min(term_count * 0.02, 0.1)
        
        final_confidence = min(base_confidence + structure_bonus + term_bonus, 1.0)
        
        return round(final_confidence, 2)
    
    async def retry_failed_analysis(self, analysis_id: int) -> bool:
        """
        Retry a failed analysis.
        
        Args:
            analysis_id (int): ID of the analysis to retry
            
        Returns:
            bool: True if retry was initiated, False otherwise
        """
        db = SessionLocal()
        
        try:
            analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
            
            if not analysis:
                return False
            
            if analysis.status != AnalysisStatus.FAILED:
                return False
            
            # Reset analysis status
            analysis.status = AnalysisStatus.PENDING
            analysis.error_message = None
            analysis.started_at = None
            analysis.completed_at = None
            
            db.commit()
            
            # Get document and agent info
            document = db.query(Document).filter(Document.id == analysis.document_id).first()
            
            if not document:
                return False
            
            # Find an available agent
            # This is a simplified implementation
            agent = db.query(Agent).filter(Agent.status == "active").first()
            
            if not agent:
                return False
            
            # Restart analysis process
            await self.process_analysis(analysis.id, agent.id, document.file_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to retry analysis {analysis_id}: {e}")
            return False
        
        finally:
            db.close()
