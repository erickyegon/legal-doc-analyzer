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
    """Comprehensive legal risk assessment with professional insights."""
    risk_counts = {"high": 0, "medium": 0, "low": 0}
    risk_factors = []
    recommendations = []

    # Count clause risks
    for clause in clauses:
        risk_level = clause.get("risk_level", "low")
        risk_counts[risk_level] += 1

    total_clauses = len(clauses)

    # Analyze specific legal risks

    # Fee and payment risks
    if re.search(r'(?i)non\s*[-\s]*transferrable\s+fee', text):
        risk_factors.append("Non-transferable fees may limit client flexibility")
        recommendations.append("Clarify fee refund policy if services are not completed")

    if re.search(r'(?i)no\s+guarantee\s+(?:of\s+)?(?:outcome|results?|success)', text):
        risk_factors.append("No guarantee of outcome clause present")
        recommendations.append("Understand that legal outcomes cannot be guaranteed")

    # Immigration-specific risks
    if re.search(r'(?i)uscis|immigration|visa|petition', text):
        risk_factors.append("Immigration matter - subject to government processing delays")
        recommendations.append("Monitor USCIS processing times and policy changes")
        recommendations.append("Ensure all supporting documentation is complete and accurate")

    # Scope limitation risks
    if re.search(r'(?i)(?:limited\s+to|does\s+not\s+include|separate\s+agreement\s+required)', text):
        risk_factors.append("Limited scope of services - additional matters may require separate agreements")
        recommendations.append("Clarify what services are and are not included")

    # Termination and withdrawal risks
    if re.search(r'(?i)terminat|withdraw', text):
        risk_factors.append("Termination provisions present")
        recommendations.append("Understand conditions under which representation may be terminated")

    # Professional responsibility risks
    if re.search(r'(?i)conflict\s+of\s+interest', text):
        risk_factors.append("Conflict of interest provisions require attention")
        recommendations.append("Disclose any potential conflicts of interest immediately")

    # Document complexity assessment
    if len(text) > 5000:
        risk_factors.append("Complex agreement - requires careful review")
        recommendations.append("Take time to thoroughly review all terms and conditions")

    # Missing critical clauses (potential risks)
    critical_clauses = {
        "confidentiality": r'(?i)confidential|privilege',
        "liability": r'(?i)liability|damages',
        "governing_law": r'(?i)governed\s+by|jurisdiction',
        "fee_structure": r'(?i)fee|payment|cost'
    }

    missing_clauses = []
    for clause_name, pattern in critical_clauses.items():
        if not re.search(pattern, text):
            missing_clauses.append(clause_name.replace('_', ' ').title())

    if missing_clauses:
        risk_factors.append(f"Missing important clauses: {', '.join(missing_clauses)}")
        recommendations.append("Consider requesting clarification on missing standard provisions")

    # Determine overall risk level
    base_risk_score = (risk_counts["high"] * 30 + risk_counts["medium"] * 15 + risk_counts["low"] * 5)
    additional_risk_score = len(risk_factors) * 5
    total_risk_score = min(100, base_risk_score + additional_risk_score)

    if total_risk_score >= 70 or risk_counts["high"] >= 3:
        overall_risk = "high"
        recommendations.insert(0, "HIGH PRIORITY: Seek immediate legal review before signing")
    elif total_risk_score >= 40 or risk_counts["high"] > 0:
        overall_risk = "medium"
        recommendations.insert(0, "RECOMMENDED: Review with legal counsel before proceeding")
    elif total_risk_score >= 20 or risk_counts["medium"] > 2:
        overall_risk = "low-medium"
        recommendations.insert(0, "SUGGESTED: Consider legal consultation for complex terms")
    else:
        overall_risk = "low"
        recommendations.insert(0, "Standard agreement - review terms carefully")

    # Add standard professional recommendations
    standard_recommendations = [
        "Ensure you understand all terms before signing",
        "Keep copies of all signed agreements and correspondence",
        "Maintain regular communication with your attorney",
        "Report any changes in circumstances that may affect your case"
    ]

    recommendations.extend(standard_recommendations)

    return {
        "overall_risk": overall_risk,
        "risk_breakdown": risk_counts,
        "total_clauses": total_clauses,
        "risk_factors": risk_factors,
        "recommendations": recommendations[:8],  # Limit to 8 most important recommendations
        "risk_score": total_risk_score,
        "risk_analysis": {
            "clause_based_risk": base_risk_score,
            "content_based_risk": additional_risk_score,
            "missing_clauses": missing_clauses,
            "document_complexity": "high" if len(text) > 5000 else "medium" if len(text) > 2000 else "low"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    logger.info(f"🚀 Starting Legal Intelligence Platform on {host}:{port}")
    logger.info(f"📋 Environment: {os.getenv('ENVIRONMENT', 'development')}")
    
    uvicorn.run(app, host=host, port=port)
