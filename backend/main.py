#!/usr/bin/env python3
"""
Legal Intelligence Platform - Production Application

This is the main FastAPI application for production deployment on Render.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import os
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import re
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Legal Intelligence Platform",
    description="AI-powered legal document analysis platform with advanced multimodal capabilities",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None
)

# Security
security = HTTPBearer(auto_error=False)

# CORS configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,https://legal-intelligence-frontend.onrender.com").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load spaCy model if available
nlp = None
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    # Increase max_length to handle larger documents (5MB of text)
    nlp.max_length = 5000000
    logger.info("✅ spaCy model loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ spaCy model not available: {e}")

# Pydantic models
class DocumentAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "comprehensive"
    include_entities: bool = True
    include_clauses: bool = True
    include_summary: bool = True
    include_risk_assessment: bool = True

class DocumentAnalysisResponse(BaseModel):
    document_id: str
    analysis_type: str
    entities: List[Dict[str, Any]]
    clauses: List[Dict[str, Any]]
    summary: str
    risk_assessment: Dict[str, Any]
    processing_time: float
    metadata: Dict[str, Any]

class ClauseSearchRequest(BaseModel):
    query: str
    clause_type: Optional[str] = None
    max_results: int = 10

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, Any]

# Enhanced Legal clause patterns for professional analysis
LEGAL_PATTERNS = {
    "professional_services": {
        "patterns": [
            r"(?i)professional\s+service\s+agreement",
            r"(?i)attorney\s+(?:will|shall)\s+provide\s+legal\s+services",
            r"(?i)scope\s+of\s+(?:services|representation)",
            r"(?i)legal\s+services\s+(?:in\s+connection\s+with|for|regarding)",
            r"(?i)retained\s+counsel"
        ],
        "description": "Professional services and legal representation clauses",
        "risk_level": "low",
        "category": "service_definition"
    },
    "fee_structure": {
        "patterns": [
            r"(?i)professional\s+fee\s*s?",
            r"(?i)(?:pay|payment\s+of)\s+(?:the\s+)?fee\s*s?",
            r"(?i)non\s*[-\s]*transferrable\s+(?:professional\s+)?fee\s*s?",
            r"(?i)retainer\s+fee",
            r"(?i)hourly\s+rate",
            r"(?i)\$[\d,]+(?:\.\d{2})?\s+(?:per\s+hour|fee|retainer)"
        ],
        "description": "Fee structure and payment terms",
        "risk_level": "high",
        "category": "financial_terms"
    },
    "conditions_precedent": {
        "patterns": [
            r"(?i)this\s+agreement\s+will\s+not\s+take\s+effect",
            r"(?i)(?:until|unless)\s+client\s+(?:returns|signs|accepts)",
            r"(?i)no\s+obligation\s+to\s+provide\s+(?:legal\s+)?services",
            r"(?i)conditions?\s+precedent",
            r"(?i)effective\s+(?:date|upon)"
        ],
        "description": "Conditions precedent and agreement effectiveness",
        "risk_level": "medium",
        "category": "contract_formation"
    },
    "immigration_services": {
        "patterns": [
            r"(?i)employment\s+based\s+(?:visa|petition)",
            r"(?i)eb[-\s]*2\s+niw",
            r"(?i)national\s+interest\s+waiver",
            r"(?i)(?:visa|petition)\s+(?:filing|preparation)",
            r"(?i)uscis\s+(?:filing|petition|application)",
            r"(?i)academic\s+evaluation"
        ],
        "description": "Immigration and visa-related services",
        "risk_level": "high",
        "category": "specialized_services"
    },
    "termination": {
        "patterns": [
            r"(?i)\b(?:terminat|end|expir|cancel|dissolv)\w*\s+(?:this\s+)?(?:agreement|contract|representation)\b",
            r"(?i)(?:upon|after|with)\s+(\d+)\s+(days?|months?|years?)\s+(?:written\s+)?notice",
            r"(?i)terminat\w*\s+(?:for\s+)?(?:cause|breach|default|material\s+breach)",
            r"(?i)withdrawal\s+of\s+(?:counsel|representation)"
        ],
        "description": "Termination and withdrawal clauses",
        "risk_level": "medium",
        "category": "contract_lifecycle"
    },
    "payment": {
        "patterns": [
            r"(?i)payment\s+(?:shall\s+be\s+)?(?:due|made|payable)\s+(?:within\s+)?(\d+)\s+(days?|months?)",
            r"(?i)(?:late\s+fee|penalty|interest)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(%|percent|dollars?)",
            r"(?i)payment\s+(?:shall\s+be\s+made\s+)?by\s+(check|wire\s+transfer|ach|credit\s+card|cash)"
        ],
        "description": "Payment terms and financial obligations",
        "risk_level": "high",
        "category": "financial"
    },
    "liability": {
        "patterns": [
            r"(?i)(?:limit|cap|maximum)\s+(?:of\s+)?liability\s+(?:shall\s+)?(?:not\s+exceed|be\s+limited\s+to)\s+\$?([\d,]+)",
            r"(?i)(?:exclud|disclaim|not\s+liable)\w*\s+(?:for\s+)?(?:any\s+)?(?:indirect|consequential|incidental|special|punitive)\s+damages",
            r"(?i)(?:indemnify|hold\s+harmless|defend)\s+(?:and\s+)?(?:hold\s+harmless|defend|indemnify)\s+(?:against|from)"
        ],
        "description": "Liability and indemnification clauses",
        "risk_level": "high",
        "category": "risk_management"
    },
    "confidentiality": {
        "patterns": [
            r"(?i)(?:confidential|proprietary|trade\s+secret)\s+information",
            r"(?i)confidentiality\s+(?:obligations?\s+)?(?:shall\s+)?(?:survive|remain\s+in\s+effect)\s+for\s+(\d+)\s+(years?|months?)",
            r"(?i)(?:non-disclosure|nda)\s+(?:agreement|provision)"
        ],
        "description": "Confidentiality and non-disclosure provisions",
        "risk_level": "medium",
        "category": "information_protection"
    },
    "intellectual_property": {
        "patterns": [
            r"(?i)(?:intellectual\s+property|copyright|patent|trademark|trade\s+secret)\s+(?:rights?\s+)?(?:shall\s+)?(?:belong\s+to|be\s+owned\s+by|vest\s+in)",
            r"(?i)(?:grant|license|permit)\s+(?:a\s+)?(?:non-exclusive|exclusive|limited|perpetual)?\s*(?:license|right)\s+to\s+use"
        ],
        "description": "Intellectual property rights and licensing",
        "risk_level": "high",
        "category": "intellectual_property"
    },
    "client_obligations": {
        "patterns": [
            r"(?i)client\s+(?:agrees|shall|must|will)\s+(?:to\s+)?(?:provide|furnish|submit)",
            r"(?i)client\s+(?:is\s+)?responsible\s+for",
            r"(?i)client\s+(?:represents|warrants|acknowledges)",
            r"(?i)cooperation\s+(?:of\s+)?(?:the\s+)?client"
        ],
        "description": "Client obligations and responsibilities",
        "risk_level": "medium",
        "category": "client_duties"
    },
    "attorney_obligations": {
        "patterns": [
            r"(?i)attorney\s+(?:agrees|shall|will)\s+(?:to\s+)?(?:provide|represent|defend)",
            r"(?i)attorney\s+(?:is\s+)?responsible\s+for",
            r"(?i)legal\s+counsel\s+(?:shall|will)",
            r"(?i)professional\s+(?:standards|conduct|ethics)"
        ],
        "description": "Attorney obligations and professional duties",
        "risk_level": "low",
        "category": "attorney_duties"
    },
    "confidentiality": {
        "patterns": [
            r"(?i)(?:confidential|privileged)\s+(?:information|communications?)",
            r"(?i)attorney[-\s]client\s+privilege",
            r"(?i)confidentiality\s+(?:obligations?\s+)?(?:shall\s+)?(?:survive|remain\s+in\s+effect)\s+for\s+(\d+)\s+(years?|months?)",
            r"(?i)(?:non-disclosure|nda)\s+(?:agreement|provision)"
        ],
        "description": "Confidentiality and privilege protections",
        "risk_level": "high",
        "category": "information_protection"
    },
    "limitation_of_liability": {
        "patterns": [
            r"(?i)(?:limit|cap|maximum)\s+(?:of\s+)?liability\s+(?:shall\s+)?(?:not\s+exceed|be\s+limited\s+to)\s+\$?([\d,]+)",
            r"(?i)(?:exclud|disclaim|not\s+liable)\w*\s+(?:for\s+)?(?:any\s+)?(?:indirect|consequential|incidental|special|punitive)\s+damages",
            r"(?i)no\s+guarantee\s+(?:of\s+)?(?:outcome|results?|success)"
        ],
        "description": "Liability limitations and disclaimers",
        "risk_level": "high",
        "category": "risk_management"
    },
    "governing_law": {
        "patterns": [
            r"(?i)(?:governed\s+by|subject\s+to|construed\s+in\s+accordance\s+with)\s+(?:the\s+)?laws?\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?i)(?:exclusive\s+)?jurisdiction\s+(?:of\s+)?(?:the\s+)?courts?\s+(?:of\s+|in\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?i)venue\s+(?:shall\s+be\s+)?(?:in\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        ],
        "description": "Governing law, jurisdiction, and venue clauses",
        "risk_level": "low",
        "category": "legal_framework"
    },
    "dispute_resolution": {
        "patterns": [
            r"(?i)(?:binding\s+)?arbitration\s+(?:in\s+accordance\s+with|under|pursuant\s+to)\s+(?:the\s+)?(?:rules\s+of\s+)?([A-Z][A-Z]+|American\s+Arbitration\s+Association)",
            r"(?i)(?:first\s+attempt\s+to\s+resolve|prior\s+to\s+litigation)\s+(?:through\s+)?mediation",
            r"(?i)dispute\s+resolution\s+(?:procedure|process|mechanism)"
        ],
        "description": "Dispute resolution mechanisms and procedures",
        "risk_level": "medium",
        "category": "dispute_resolution"
    },
    "scope_limitations": {
        "patterns": [
            r"(?i)(?:limited\s+to|restricted\s+to|only\s+includes?)\s+(?:the\s+)?(?:following|services)",
            r"(?i)does\s+not\s+include",
            r"(?i)(?:excludes?|excluding)\s+(?:any\s+)?(?:other|additional)\s+(?:services|matters)",
            r"(?i)separate\s+(?:agreement|retainer)\s+(?:required|necessary)"
        ],
        "description": "Scope limitations and service boundaries",
        "risk_level": "medium",
        "category": "service_definition"
    }
}

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Legal Intelligence Platform API",
        "version": "1.0.0",
        "status": "running",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "features": [
            "Advanced document analysis",
            "Multimodal entity extraction",
            "Legal clause detection",
            "Risk assessment",
            "Legal summarization",
            "Clause library search",
            "Real-time processing"
        ],
        "endpoints": {
            "health": "/health",
            "analyze": "/api/v1/analyze",
            "upload": "/api/v1/upload",
            "search_clauses": "/api/v1/search-clauses"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        services={
            "spacy_nlp": nlp is not None,
            "regex_patterns": len(LEGAL_PATTERNS),
            "api_server": True,
            "database": False,  # Will be True when DB is connected
            "ai_models": {
                "spacy": nlp is not None,
                "euri_client": True
            }
        }
    )

@app.post("/api/v1/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(request: DocumentAnalysisRequest):
    """
    Comprehensive legal document analysis.
    
    This endpoint provides:
    - Named entity extraction
    - Legal clause detection
    - Risk assessment
    - Document summarization
    """
    start_time = datetime.now()
    
    try:
        # Generate document ID
        doc_id = f"doc_{hash(request.text) % 1000000:06d}"
        
        # Initialize results
        entities = []
        clauses = []
        summary = ""
        risk_assessment = {}
        
        # Extract entities if requested
        if request.include_entities:
            entities = extract_entities(request.text)
        
        # Detect legal clauses if requested
        if request.include_clauses:
            clauses = detect_clauses(request.text)
        
        # Generate summary if requested
        if request.include_summary:
            summary = generate_summary(request.text)
        
        # Assess risks if requested
        if request.include_risk_assessment:
            risk_assessment = assess_risks(clauses, request.text)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Generate metadata
        metadata = {
            "document_length": len(request.text),
            "word_count": len(request.text.split()),
            "sentence_count": request.text.count('.'),
            "analysis_timestamp": datetime.now().isoformat(),
            "processing_method": "advanced_nlp" if nlp else "regex_based"
        }
        
        return DocumentAnalysisResponse(
            document_id=doc_id,
            analysis_type=request.analysis_type,
            entities=entities,
            clauses=clauses,
            summary=summary,
            risk_assessment=risk_assessment,
            processing_time=processing_time,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Document analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/search-clauses")
async def search_clauses(request: ClauseSearchRequest):
    """Search for similar clauses in the clause library."""
    try:
        results = []
        query_lower = request.query.lower()
        
        for clause_type, pattern_info in LEGAL_PATTERNS.items():
            if request.clause_type and request.clause_type != clause_type:
                continue
                
            # Calculate relevance score
            relevance_score = 0.0
            
            # Check if query matches clause type
            if clause_type.replace('_', ' ') in query_lower:
                relevance_score += 0.8
            
            # Check if query matches description
            if any(word in query_lower for word in pattern_info["description"].lower().split()):
                relevance_score += 0.6
            
            # Check if query matches category
            if pattern_info["category"].replace('_', ' ') in query_lower:
                relevance_score += 0.4
            
            if relevance_score > 0.3:  # Minimum relevance threshold
                results.append({
                    "clause_type": clause_type,
                    "description": pattern_info["description"],
                    "risk_level": pattern_info["risk_level"],
                    "category": pattern_info["category"],
                    "similarity_score": min(relevance_score, 1.0),
                    "pattern_count": len(pattern_info["patterns"]),
                    "sample_patterns": pattern_info["patterns"][:2]  # Show first 2 patterns
                })
        
        # Sort by similarity score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "query": request.query,
            "results": results[:request.max_results],
            "total_results": len(results),
            "search_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Clause search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/v1/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and analyze a document with real PDF text extraction."""
    try:
        # Validate file type
        allowed_types = ["text/plain", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Supported types: {', '.join(allowed_types)}"
            )

        # Validate file size (50MB limit)
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 50MB limit"
            )

        # Extract text based on file type
        extracted_text = ""
        extraction_method = "unknown"

        if file.content_type == "text/plain":
            extracted_text = content.decode("utf-8")
            extraction_method = "direct_text"

        elif file.content_type == "application/pdf":
            # Real PDF text extraction using PyPDF2
            try:
                import PyPDF2
                import io

                pdf_file = io.BytesIO(content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                text_parts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        text_parts.append(f"--- Page {page_num + 1} ---\n[Text extraction failed for this page]")

                if text_parts:
                    extracted_text = "\n\n".join(text_parts)
                    extraction_method = "pypdf2"
                else:
                    extracted_text = "[No text could be extracted from this PDF. The PDF might be image-based or encrypted.]"
                    extraction_method = "pypdf2_failed"

            except ImportError:
                extracted_text = f"[PDF Processing] PDF text extraction from {file.filename}. PyPDF2 not available - using mock extraction."
                extraction_method = "mock_pdf"
            except Exception as e:
                logger.error(f"PDF extraction failed: {e}")
                extracted_text = f"[PDF Processing Error] Failed to extract text from {file.filename}. Error: {str(e)}"
                extraction_method = "pypdf2_error"

        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # DOCX text extraction
            try:
                import docx
                import io

                docx_file = io.BytesIO(content)
                doc = docx.Document(docx_file)

                text_parts = []
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        text_parts.append(paragraph.text)

                extracted_text = "\n".join(text_parts) if text_parts else "[No text found in DOCX document]"
                extraction_method = "python_docx"

            except ImportError:
                extracted_text = f"[DOCX Processing] DOCX text extraction from {file.filename}. python-docx not available - using mock extraction."
                extraction_method = "mock_docx"
            except Exception as e:
                logger.error(f"DOCX extraction failed: {e}")
                extracted_text = f"[DOCX Processing Error] Failed to extract text from {file.filename}. Error: {str(e)}"
                extraction_method = "docx_error"
        else:
            extracted_text = f"[Document Content] Unsupported file type for text extraction: {file.content_type}"
            extraction_method = "unsupported"

        # Ensure we have some text to analyze
        if not extracted_text.strip():
            extracted_text = f"[Empty Document] No text content found in {file.filename}"

        # Validate text length for processing
        if len(extracted_text) > 2000000:  # 2MB text limit
            # Truncate very long texts
            extracted_text = extracted_text[:2000000] + "\n\n[Document truncated due to length - showing first 2MB of text]"
            logger.warning(f"Document {file.filename} truncated due to length: {len(extracted_text)} characters")

        # Analyze the extracted text
        analysis_request = DocumentAnalysisRequest(text=extracted_text)
        result = await analyze_document(analysis_request)

        # Add extraction metadata to the result
        result.metadata.update({
            "extracted_text": extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text,
            "extraction_method": extraction_method,
            "original_filename": file.filename,
            "file_size_bytes": len(content),
            "pages_processed": len(extracted_text.split("--- Page")) - 1 if "--- Page" in extracted_text else 1
        })

        return {
            "upload_info": {
                "filename": file.filename,
                "file_size": len(content),
                "content_type": file.content_type,
                "upload_timestamp": datetime.now().isoformat(),
                "extraction_method": extraction_method,
                "text_length": len(extracted_text),
                "pages_processed": result.metadata.get("pages_processed", 1) if hasattr(result, 'metadata') else 1
            },
            "analysis": result.dict() if hasattr(result, 'dict') else result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract meaningful legal entities from text using enhanced patterns."""
    entities = []

    # Enhanced legal entity patterns
    legal_patterns = {
        "LAW_FIRM": {
            "pattern": r'\b[A-Z][a-z]+(?:\s+[&]\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*,?\s+(?:P\.?L\.?|PLLC|LLP|LLC|PC|P\.A\.)\b',
            "description": "Law firm or legal entity"
        },
        "PERSON_NAME": {
            "pattern": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b(?=\s*\((?:hereinafter|Client|Attorney))',
            "description": "Person name (client or attorney)"
        },
        "LEGAL_DOCUMENT": {
            "pattern": r'(?i)\b(?:agreement|contract|petition|application|motion|brief|pleading|complaint|answer)\b',
            "description": "Legal document type"
        },
        "VISA_TYPE": {
            "pattern": r'(?i)\b(?:EB[-\s]*[12345]|H[-\s]*1B|L[-\s]*1|O[-\s]*1|NIW|PERM|I[-\s]*\d+)\b',
            "description": "Immigration visa or form type"
        },
        "MONEY_AMOUNT": {
            "pattern": r'\$[\d,]+(?:\.\d{2})?',
            "description": "Monetary amount"
        },
        "PHONE_NUMBER": {
            "pattern": r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            "description": "Phone number"
        },
        "ADDRESS": {
            "pattern": r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Suite|Ste)\.?\s*\d*',
            "description": "Street address"
        },
        "STATE_ZIP": {
            "pattern": r'\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b',
            "description": "State and ZIP code"
        },
        "DATE_FORMAL": {
            "pattern": r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            "description": "Formal date"
        },
        "TIME_PERIOD": {
            "pattern": r'\b\d+\s+(?:days?|months?|years?|weeks?)\b',
            "description": "Time period or duration"
        },
        "PROFESSIONAL_TITLE": {
            "pattern": r'(?i)\b(?:attorney|counsel|lawyer|esquire|esq\.?|partner|associate)\b',
            "description": "Professional legal title"
        },
        "COURT_REFERENCE": {
            "pattern": r'(?i)\b(?:court|tribunal|judge|magistrate|jurisdiction)\b',
            "description": "Court or judicial reference"
        }
    }

    # First try spaCy if available for standard entities
    if nlp:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                # Filter for meaningful legal entities
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'MONEY', 'DATE', 'TIME', 'LAW']:
                    entities.append({
                        "text": ent.text.strip(),
                        "label": ent.label_,
                        "description": spacy.explain(ent.label_) if hasattr(spacy, 'explain') else ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.9,
                        "source": "spacy"
                    })
        except Exception as e:
            logger.warning(f"spaCy entity extraction failed: {e}")

    # Add legal-specific pattern matching
    for label, pattern_info in legal_patterns.items():
        for match in re.finditer(pattern_info["pattern"], text):
            matched_text = match.group().strip()
            if len(matched_text) > 1:  # Filter out single characters
                entities.append({
                    "text": matched_text,
                    "label": label,
                    "description": pattern_info["description"],
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.85,
                    "source": "legal_patterns"
                })

    # Remove duplicates and sort by position
    unique_entities = []
    seen_texts = set()

    for entity in sorted(entities, key=lambda x: x["start"]):
        # Avoid duplicate entities (same text and similar position)
        entity_key = (entity["text"].lower(), entity["start"] // 10)  # Group by 10-char windows
        if entity_key not in seen_texts:
            seen_texts.add(entity_key)
            unique_entities.append(entity)

    return unique_entities[:50]  # Limit to top 50 entities

def detect_clauses(text: str) -> List[Dict[str, Any]]:
    """Detect legal clauses in text using pattern matching."""
    clauses = []
    
    for clause_type, pattern_info in LEGAL_PATTERNS.items():
        for i, pattern in enumerate(pattern_info["patterns"]):
            for match in re.finditer(pattern, text):
                clauses.append({
                    "clause_type": clause_type,
                    "text": match.group(),
                    "description": pattern_info["description"],
                    "risk_level": pattern_info["risk_level"],
                    "category": pattern_info["category"],
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.85,
                    "pattern_index": i
                })
    
    return clauses

def generate_summary(text: str) -> str:
    """Generate a meaningful legal document summary."""

    # Identify document type
    doc_type = "Legal Document"
    if re.search(r'(?i)professional\s+service\s+agreement', text):
        doc_type = "Professional Services Agreement"
    elif re.search(r'(?i)employment\s+agreement', text):
        doc_type = "Employment Agreement"
    elif re.search(r'(?i)license\s+agreement', text):
        doc_type = "License Agreement"
    elif re.search(r'(?i)retainer\s+agreement', text):
        doc_type = "Retainer Agreement"

    # Extract key information
    summary_parts = [f"Document Type: {doc_type}"]

    # Extract parties
    parties = []
    party_patterns = [
        r'(?i)between\s+([^,]+(?:,\s*[^,]+)*)\s+(?:\([^)]*\))?\s+and\s+([^,]+(?:,\s*[^,]+)*)',
        r'(?i)client["\s]*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'(?i)attorney["\s]*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z\.]+)*)'
    ]

    for pattern in party_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            if len(match.groups()) >= 2:
                parties.extend([match.group(1).strip(), match.group(2).strip()])
            else:
                parties.append(match.group(1).strip())

    if parties:
        unique_parties = list(dict.fromkeys(parties))[:3]  # Remove duplicates, limit to 3
        summary_parts.append(f"Parties: {', '.join(unique_parties)}")

    # Extract services
    services = []
    service_patterns = [
        r'(?i)(?:provide|preparation|filing)\s+(?:of\s+)?([^.]+(?:visa|petition|application|evaluation)[^.]*)',
        r'(?i)legal\s+services\s+(?:in\s+connection\s+with|for|regarding)\s+([^.]+)',
        r'(?i)scope\s+of\s+services[^:]*:\s*([^.]+)'
    ]

    for pattern in service_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            service = match.group(1).strip()
            if len(service) > 10 and len(service) < 200:
                services.append(service)

    if services:
        summary_parts.append(f"Services: {services[0]}")

    # Extract key terms
    key_terms = []

    # Look for fees
    fee_matches = re.finditer(r'(?i)(?:fee|payment|retainer)[^.]*\$[\d,]+(?:\.\d{2})?[^.]*', text)
    for match in fee_matches:
        fee_text = match.group().strip()
        if len(fee_text) < 150:
            key_terms.append(f"Fee: {fee_text}")
            break

    # Look for important conditions
    condition_matches = re.finditer(r'(?i)(?:this\s+agreement\s+will\s+not\s+take\s+effect|conditions?)[^.]*[^.]{10,100}', text)
    for match in condition_matches:
        condition = match.group().strip()
        if len(condition) < 150:
            key_terms.append(f"Condition: {condition}")
            break

    # Look for termination terms
    termination_matches = re.finditer(r'(?i)(?:terminat|withdraw)[^.]*[^.]{10,100}', text)
    for match in termination_matches:
        termination = match.group().strip()
        if len(termination) < 150:
            key_terms.append(f"Termination: {termination}")
            break

    if key_terms:
        summary_parts.extend(key_terms[:3])  # Limit to 3 key terms

    # Extract important dates or deadlines
    date_matches = re.finditer(r'(?i)(?:within|after|before)\s+\d+\s+(?:days?|months?|years?)[^.]*', text)
    for match in date_matches:
        date_term = match.group().strip()
        if len(date_term) < 100:
            summary_parts.append(f"Timeline: {date_term}")
            break

    # Combine summary
    if len(summary_parts) > 1:
        summary = ". ".join(summary_parts) + "."
    else:
        # Fallback to extractive summary
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 20]
        if sentences:
            summary = f"{doc_type}: " + ". ".join(sentences[:3]) + "."
        else:
            summary = f"{doc_type}: Document analysis completed."

    # Ensure reasonable length
    if len(summary) > 800:
        summary = summary[:800] + "..."

    return summary

def assess_risks(clauses: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
    """Professional legal risk assessment with detailed scoring and explanations."""
    risk_counts = {"high": 0, "medium": 0, "low": 0}
    risk_factors = []
    recommendations = []
    scoring_details = []

    # Count clause risks
    for clause in clauses:
        risk_level = clause.get("risk_level", "low")
        risk_counts[risk_level] += 1

    total_clauses = len(clauses)

    # Initialize professional scoring system (0-100 scale)
    completeness_score = 50  # Base score
    clarity_score = 50
    protection_score = 50
    compliance_score = 50
    enforceability_score = 50
    scoring_details = []

    # COMPLETENESS ANALYSIS (Critical legal elements present)
    required_elements = {
        "parties_identification": r'(?i)(?:between|client|attorney|party)',
        "scope_of_services": r'(?i)(?:scope|services|representation|legal services)',
        "fee_structure": r'(?i)(?:fee|payment|cost|retainer)',
        "termination_clause": r'(?i)(?:terminat|end|withdraw|cancel)',
        "governing_law": r'(?i)(?:governed|jurisdiction|law)',
        "effective_date": r'(?i)(?:effective|commence|begin|date)',
        "signatures": r'(?i)(?:sign|execute|agreement)',
        "conditions": r'(?i)(?:condition|requirement|obligation)'
    }

    present_elements = 0
    missing_elements = []

    for element, pattern in required_elements.items():
        if re.search(pattern, text):
            present_elements += 1
            completeness_score += 6  # +6 points per element (max 48 additional)
        else:
            missing_elements.append(element.replace('_', ' ').title())

    completeness_percentage = min(100, completeness_score)
    scoring_details.append({
        "category": "Document Completeness",
        "score": completeness_percentage,
        "explanation": f"Document contains {present_elements}/{len(required_elements)} essential legal elements. " +
                      (f"Missing: {', '.join(missing_elements)}" if missing_elements else "All key elements present."),
        "impact": "high" if completeness_percentage < 60 else "medium" if completeness_percentage < 80 else "low"
    })

    # CLARITY AND SPECIFICITY ANALYSIS
    clarity_factors = {
        "specific_amounts": r'\$[\d,]+(?:\.\d{2})?',
        "specific_dates": r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        "specific_timeframes": r'\b\d+\s+(?:days?|months?|years?|weeks?)\b',
        "defined_terms": r'\([^)]*hereinafter[^)]*\)|"[^"]*"',
        "numbered_sections": r'\b\d+\.\s+[A-Z]',
        "contact_information": r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}|[\w\.-]+@[\w\.-]+\.\w+'
    }

    clarity_points = 0
    clarity_explanations = []

    for factor, pattern in clarity_factors.items():
        matches = len(re.findall(pattern, text))
        if matches > 0:
            points = min(8, matches * 2)  # Max 8 points per factor
            clarity_points += points
            clarity_explanations.append(f"{factor.replace('_', ' ').title()}: {matches} instances (+{points} pts)")

    clarity_score = min(100, 30 + clarity_points)  # Base 30 + up to 70 from factors
    scoring_details.append({
        "category": "Clarity & Specificity",
        "score": clarity_score,
        "explanation": f"Document specificity analysis: {'; '.join(clarity_explanations) if clarity_explanations else 'Limited specific details found'}",
        "impact": "high" if clarity_score < 50 else "medium" if clarity_score < 75 else "low"
    })

    # CLIENT PROTECTION ANALYSIS
    protection_factors = {
        "fee_transparency": r'(?i)(?:fee|cost|expense)\s+(?:schedule|structure|breakdown)',
        "refund_policy": r'(?i)(?:refund|return|reimburse)',
        "liability_limitation": r'(?i)(?:limit|cap|maximum)\s+(?:of\s+)?liability',
        "confidentiality": r'(?i)(?:confidential|privilege|private)',
        "communication_rights": r'(?i)(?:inform|notify|update|communicate)',
        "termination_rights": r'(?i)(?:terminat|withdraw|end)\s+(?:representation|agreement)',
        "dispute_resolution": r'(?i)(?:arbitration|mediation|dispute\s+resolution)'
    }

    protection_points = 0
    protection_explanations = []

    for factor, pattern in protection_factors.items():
        if re.search(pattern, text):
            protection_points += 10
            protection_explanations.append(f"{factor.replace('_', ' ').title()}: Present")
        else:
            protection_explanations.append(f"{factor.replace('_', ' ').title()}: Missing")

    protection_score = min(100, 30 + protection_points)  # Base 30 + up to 70
    scoring_details.append({
        "category": "Client Protection",
        "score": protection_score,
        "explanation": f"Client protection measures: {'; '.join(protection_explanations)}",
        "impact": "high" if protection_score < 60 else "medium" if protection_score < 80 else "low"
    })

    # COMPLIANCE AND PROFESSIONAL STANDARDS ANALYSIS
    compliance_factors = {
        "professional_rules": r'(?i)(?:professional|ethical|rules|standards)',
        "bar_requirements": r'(?i)(?:bar|license|admission)',
        "conflict_disclosure": r'(?i)(?:conflict|interest|disclosure)',
        "client_funds": r'(?i)(?:trust|escrow|client\s+funds)',
        "record_keeping": r'(?i)(?:record|file|document|maintain)',
        "supervision": r'(?i)(?:supervis|oversight|review)'
    }

    compliance_points = 0
    compliance_explanations = []

    for factor, pattern in compliance_factors.items():
        if re.search(pattern, text):
            compliance_points += 12
            compliance_explanations.append(f"{factor.replace('_', ' ').title()}: Addressed")

    compliance_score = min(100, 28 + compliance_points)  # Base 28 + up to 72
    scoring_details.append({
        "category": "Professional Compliance",
        "score": compliance_score,
        "explanation": f"Professional standards compliance: {'; '.join(compliance_explanations) if compliance_explanations else 'Limited compliance provisions found'}",
        "impact": "high" if compliance_score < 50 else "medium" if compliance_score < 75 else "low"
    })

    # ENFORCEABILITY ANALYSIS
    enforceability_factors = {
        "consideration": r'(?i)(?:consideration|payment|fee|exchange)',
        "mutual_obligations": r'(?i)(?:both\s+parties|mutual|reciprocal)',
        "legal_capacity": r'(?i)(?:capacity|authority|power)',
        "lawful_purpose": r'(?i)(?:legal|lawful|legitimate)',
        "written_agreement": r'(?i)(?:written|agreement|contract)',
        "proper_execution": r'(?i)(?:sign|execute|witness|notari)'
    }

    enforceability_points = 0
    enforceability_explanations = []

    for factor, pattern in enforceability_factors.items():
        if re.search(pattern, text):
            enforceability_points += 12
            enforceability_explanations.append(f"{factor.replace('_', ' ').title()}: Present")

    enforceability_score = min(100, 28 + enforceability_points)
    scoring_details.append({
        "category": "Legal Enforceability",
        "score": enforceability_score,
        "explanation": f"Enforceability factors: {'; '.join(enforceability_explanations) if enforceability_explanations else 'Basic enforceability elements present'}",
        "impact": "high" if enforceability_score < 60 else "medium" if enforceability_score < 80 else "low"
    })

    # SPECIFIC RISK ANALYSIS WITH DETAILED EXPLANATIONS
    # Fee and payment risks
    if re.search(r'(?i)non\s*[-\s]*transferrable\s+fee', text):
        risk_factors.append("Non-transferable fees may limit client flexibility and create financial risk if representation is terminated early")
        recommendations.append("Request clarification on fee refund policy if services are not completed or representation is terminated")

    if re.search(r'(?i)no\s+guarantee\s+(?:of\s+)?(?:outcome|results?|success)', text):
        risk_factors.append("No guarantee of outcome clause present - standard but important for client expectations")
        recommendations.append("Understand that legal outcomes cannot be guaranteed, but ensure clear communication about case strategy and progress")

    # IMMIGRATION-SPECIFIC PROFESSIONAL ANALYSIS
    if re.search(r'(?i)uscis|immigration|visa|petition', text):
        risk_factors.append("Immigration matter - subject to federal agency processing delays, policy changes, and complex regulatory requirements")
        recommendations.append("Monitor USCIS processing times and policy changes that may affect case timeline and strategy")
        recommendations.append("Ensure all supporting documentation meets current USCIS requirements and is properly authenticated")
        recommendations.append("Maintain backup documentation and prepare for potential Requests for Evidence (RFEs)")

        # Immigration-specific scoring adjustments
        if re.search(r'(?i)eb[-\s]*2|niw|national\s+interest\s+waiver', text):
            risk_factors.append("EB-2 NIW petition involves complex legal standards requiring demonstration of national interest")
            recommendations.append("Ensure comprehensive documentation of achievements, impact, and national interest benefit")

    # SCOPE LIMITATION PROFESSIONAL ANALYSIS
    if re.search(r'(?i)(?:limited\s+to|does\s+not\s+include|separate\s+agreement\s+required)', text):
        risk_factors.append("Limited scope of services creates potential for additional costs and separate retainer agreements for related matters")
        recommendations.append("Request detailed written clarification of what services are included and excluded from this agreement")
        recommendations.append("Discuss potential related legal matters that may arise and associated costs")

    # TERMINATION AND WITHDRAWAL PROFESSIONAL ANALYSIS
    if re.search(r'(?i)terminat|withdraw', text):
        risk_factors.append("Termination provisions present - important for understanding circumstances under which representation may end")
        recommendations.append("Understand specific conditions under which attorney may withdraw and client obligations upon termination")
        recommendations.append("Clarify status of work product and files upon termination of representation")

    # PROFESSIONAL RESPONSIBILITY ANALYSIS
    if re.search(r'(?i)conflict\s+of\s+interest', text):
        risk_factors.append("Conflict of interest provisions require ongoing attention throughout representation")
        recommendations.append("Disclose any potential conflicts of interest immediately and throughout the representation")

    # DOCUMENT COMPLETENESS ASSESSMENT
    if len(text) < 1000:
        risk_factors.append("Document appears incomplete - critical terms and conditions may be missing")
        recommendations.append("Request complete agreement with all exhibits, schedules, and attachments")
        recommendations.append("Verify this is the final, complete version of the agreement")
    elif len(text) > 5000:
        risk_factors.append("Complex agreement with extensive terms - requires thorough professional review")
        recommendations.append("Allow adequate time for comprehensive review of all terms and conditions")
        recommendations.append("Consider consulting with specialized counsel for complex provisions")

    # CRITICAL CLAUSE ANALYSIS WITH PROFESSIONAL IMPLICATIONS
    critical_clauses = {
        "confidentiality": {
            "pattern": r'(?i)confidential|privilege',
            "risk": "Lack of confidentiality provisions may compromise attorney-client privilege protection"
        },
        "liability": {
            "pattern": r'(?i)liability|damages|malpractice',
            "risk": "Absence of liability provisions creates uncertainty about professional responsibility limits"
        },
        "governing_law": {
            "pattern": r'(?i)governed\s+by|jurisdiction|venue',
            "risk": "Missing governing law clause creates uncertainty about applicable legal standards and forum"
        },
        "fee_structure": {
            "pattern": r'(?i)fee\s+(?:schedule|structure|breakdown)|hourly\s+rate|flat\s+fee',
            "risk": "Unclear fee structure may lead to billing disputes and unexpected costs"
        },
        "communication": {
            "pattern": r'(?i)communicat|report|update|inform',
            "risk": "Lack of communication provisions may result in inadequate client updates"
        }
    }

    missing_clauses = []
    for clause_name, clause_info in critical_clauses.items():
        if not re.search(clause_info["pattern"], text):
            missing_clauses.append(clause_name.replace('_', ' ').title())
            risk_factors.append(f"Missing {clause_name.replace('_', ' ')} provisions: {clause_info['risk']}")

    if missing_clauses:
        recommendations.append(f"Request addition of missing critical provisions: {', '.join(missing_clauses)}")
        recommendations.append("Standard professional service agreements should include comprehensive protection clauses")

    # PROFESSIONAL OVERALL RISK ASSESSMENT WITH DETAILED SCORING

    # Calculate composite professional score
    overall_professional_score = (
        completeness_percentage * 0.25 +  # 25% weight
        clarity_score * 0.20 +            # 20% weight
        protection_score * 0.25 +         # 25% weight
        compliance_score * 0.15 +         # 15% weight
        enforceability_score * 0.15       # 15% weight
    )

    # Risk level determination based on professional standards
    if overall_professional_score >= 85:
        overall_risk = "low"
        risk_level_explanation = "Excellent professional agreement meeting high legal standards"
        priority_action = "APPROVED: Well-drafted agreement suitable for execution"
    elif overall_professional_score >= 70:
        overall_risk = "low-medium"
        risk_level_explanation = "Good professional agreement with minor areas for improvement"
        priority_action = "RECOMMENDED: Minor revisions suggested before execution"
    elif overall_professional_score >= 55:
        overall_risk = "medium"
        risk_level_explanation = "Adequate agreement with several areas requiring attention"
        priority_action = "CAUTION: Significant revisions recommended before execution"
    elif overall_professional_score >= 40:
        overall_risk = "medium-high"
        risk_level_explanation = "Below-standard agreement with multiple deficiencies"
        priority_action = "WARNING: Major revisions required before execution"
    else:
        overall_risk = "high"
        risk_level_explanation = "Inadequate agreement failing to meet professional standards"
        priority_action = "CRITICAL: Complete revision required - do not execute in current form"

    # Additional risk factors based on clause analysis
    base_risk_score = (risk_counts["high"] * 30 + risk_counts["medium"] * 15 + risk_counts["low"] * 5)

    # Professional recommendations based on scoring
    professional_recommendations = [priority_action]

    # Add specific recommendations based on scoring categories
    if completeness_percentage < 70:
        professional_recommendations.append("CRITICAL: Address missing essential legal elements before proceeding")
    if clarity_score < 60:
        professional_recommendations.append("IMPORTANT: Request clarification of vague or ambiguous terms")
    if protection_score < 60:
        professional_recommendations.append("ESSENTIAL: Strengthen client protection provisions")
    if compliance_score < 60:
        professional_recommendations.append("REQUIRED: Ensure compliance with professional standards")
    if enforceability_score < 60:
        professional_recommendations.append("NECESSARY: Address enforceability concerns")

    # Add context-specific recommendations
    professional_recommendations.extend(recommendations[:5])  # Include specific risk recommendations

    # Standard professional practice recommendations
    standard_professional_recommendations = [
        "Maintain detailed records of all communications and decisions",
        "Ensure compliance with applicable state bar rules and regulations",
        "Review agreement periodically for changes in law or circumstances",
        "Seek second opinion for complex or unusual provisions"
    ]

    professional_recommendations.extend(standard_professional_recommendations)

    # INCOMPLETE DOCUMENT ANALYSIS
    document_status = "complete"
    completeness_issues = []

    if len(text) < 500:
        document_status = "severely_incomplete"
        completeness_issues.append("Document appears to be a fragment - missing substantial content")
    elif len(text) < 1500:
        document_status = "incomplete"
        completeness_issues.append("Document appears incomplete - may be missing pages or sections")
    elif "..." in text or text.endswith("r..."):
        document_status = "truncated"
        completeness_issues.append("Document appears to be truncated - content cut off")

    if re.search(r'(?i)see\s+(?:exhibit|attachment|schedule|appendix)', text):
        completeness_issues.append("References to external documents that are not included")

    if not re.search(r'(?i)(?:signature|sign|execute)', text):
        completeness_issues.append("No signature provisions found - may be missing signature page")

    return {
        "overall_risk": overall_risk,
        "risk_breakdown": risk_counts,
        "total_clauses": total_clauses,
        "risk_factors": risk_factors,
        "recommendations": professional_recommendations[:12],  # Top 12 recommendations
        "risk_score": int(overall_professional_score),
        "professional_analysis": {
            "overall_score": round(overall_professional_score, 1),
            "risk_level_explanation": risk_level_explanation,
            "priority_action": priority_action,
            "scoring_breakdown": scoring_details,
            "document_status": document_status,
            "completeness_issues": completeness_issues,
            "professional_standards_met": overall_professional_score >= 70,
            "requires_legal_review": overall_professional_score < 55,
            "execution_recommended": overall_professional_score >= 85
        },
        "detailed_scores": {
            "completeness": round(completeness_percentage, 1),
            "clarity": round(clarity_score, 1),
            "client_protection": round(protection_score, 1),
            "compliance": round(compliance_score, 1),
            "enforceability": round(enforceability_score, 1)
        },
        "risk_analysis": {
            "clause_based_risk": base_risk_score,
            "content_based_risk": len(risk_factors) * 5,
            "missing_clauses": missing_clauses,
            "document_complexity": "high" if len(text) > 5000 else "medium" if len(text) > 2000 else "low",
            "professional_grade": "excellent" if overall_professional_score >= 85 else
                                 "good" if overall_professional_score >= 70 else
                                 "adequate" if overall_professional_score >= 55 else
                                 "poor" if overall_professional_score >= 40 else "inadequate"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    logger.info(f"🚀 Starting Legal Intelligence Platform on {host}:{port}")
    logger.info(f"📋 Environment: {os.getenv('ENVIRONMENT', 'development')}")
    
    uvicorn.run(app, host=host, port=port)
