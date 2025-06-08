#!/usr/bin/env python3
"""
Demo server for Legal Intelligence Platform.

This is a simplified FastAPI server that demonstrates the core functionality
without complex dependencies.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import re
import json
from datetime import datetime
import spacy

# Initialize FastAPI app
app = FastAPI(
    title="Legal Intelligence Platform",
    description="AI-powered legal document analysis platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully")
except Exception as e:
    print(f"⚠️ spaCy model not available: {e}")
    nlp = None

# Pydantic models
class DocumentAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "comprehensive"

class DocumentAnalysisResponse(BaseModel):
    document_id: str
    analysis_type: str
    entities: List[Dict[str, Any]]
    clauses: List[Dict[str, Any]]
    summary: str
    risk_assessment: Dict[str, Any]
    processing_time: float

class ClauseSearchRequest(BaseModel):
    query: str
    clause_type: Optional[str] = None

# Legal clause patterns
LEGAL_PATTERNS = {
    "termination": {
        "pattern": r"(?i)\b(?:terminat|end|expir|cancel|dissolv)\w*\s+(?:this\s+)?(?:agreement|contract|lease)\b",
        "description": "Termination clauses",
        "risk_level": "medium"
    },
    "payment": {
        "pattern": r"(?i)payment\s+(?:shall\s+be\s+)?(?:due|made|payable)\s+(?:within\s+)?(\d+)\s+(days?|months?)",
        "description": "Payment terms",
        "risk_level": "high"
    },
    "liability": {
        "pattern": r"(?i)(?:limit|cap|maximum)\s+(?:of\s+)?liability\s+(?:shall\s+)?(?:not\s+exceed|be\s+limited\s+to)",
        "description": "Liability limitation",
        "risk_level": "high"
    },
    "confidentiality": {
        "pattern": r"(?i)(?:confidential|proprietary|trade\s+secret)\s+information",
        "description": "Confidentiality clauses",
        "risk_level": "medium"
    },
    "governing_law": {
        "pattern": r"(?i)(?:governed\s+by|subject\s+to|construed\s+in\s+accordance\s+with)\s+(?:the\s+)?laws?\s+of\s+([A-Z][a-z]+)",
        "description": "Governing law clauses",
        "risk_level": "low"
    }
}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Legal Intelligence Platform API",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "Document analysis",
            "Entity extraction",
            "Clause detection",
            "Risk assessment",
            "Legal summarization"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "spacy_nlp": nlp is not None,
            "regex_patterns": len(LEGAL_PATTERNS),
            "api_server": True
        }
    }

@app.post("/api/v1/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(request: DocumentAnalysisRequest):
    """Analyze a legal document."""
    start_time = datetime.now()
    
    try:
        # Generate document ID
        doc_id = f"doc_{hash(request.text) % 1000000:06d}"
        
        # Extract entities using spaCy
        entities = extract_entities(request.text)
        
        # Detect legal clauses
        clauses = detect_clauses(request.text)
        
        # Generate summary
        summary = generate_summary(request.text)
        
        # Assess risks
        risk_assessment = assess_risks(clauses)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DocumentAnalysisResponse(
            document_id=doc_id,
            analysis_type=request.analysis_type,
            entities=entities,
            clauses=clauses,
            summary=summary,
            risk_assessment=risk_assessment,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/search-clauses")
async def search_clauses(request: ClauseSearchRequest):
    """Search for similar clauses."""
    try:
        # Simple clause search based on keywords
        results = []
        
        query_lower = request.query.lower()
        
        for clause_type, pattern_info in LEGAL_PATTERNS.items():
            if request.clause_type and request.clause_type != clause_type:
                continue
                
            # Simple keyword matching
            if any(keyword in query_lower for keyword in clause_type.split('_')):
                results.append({
                    "clause_type": clause_type,
                    "description": pattern_info["description"],
                    "risk_level": pattern_info["risk_level"],
                    "similarity_score": 0.8,  # Mock similarity score
                    "sample_text": f"Sample {pattern_info['description'].lower()} clause text..."
                })
        
        return {
            "query": request.query,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/v1/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and analyze a document."""
    try:
        # Read file content
        content = await file.read()
        
        # For demo, assume text files
        if file.content_type == "text/plain":
            text = content.decode("utf-8")
        else:
            # For other file types, return a mock response
            text = f"Mock extracted text from {file.filename}"
        
        # Analyze the document
        analysis_request = DocumentAnalysisRequest(text=text)
        result = await analyze_document(analysis_request)
        
        return {
            "filename": file.filename,
            "file_size": len(content),
            "content_type": file.content_type,
            "analysis": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract named entities from text."""
    entities = []
    
    if nlp:
        # Use spaCy for entity extraction
        doc = nlp(text)
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_),
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 0.9  # Mock confidence score
            })
    else:
        # Fallback regex-based entity extraction
        # Money amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        for match in re.finditer(money_pattern, text):
            entities.append({
                "text": match.group(),
                "label": "MONEY",
                "description": "Monetary values",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.8
            })
        
        # Dates
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        for match in re.finditer(date_pattern, text):
            entities.append({
                "text": match.group(),
                "label": "DATE",
                "description": "Dates",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.8
            })
    
    return entities

def detect_clauses(text: str) -> List[Dict[str, Any]]:
    """Detect legal clauses in text."""
    clauses = []
    
    for clause_type, pattern_info in LEGAL_PATTERNS.items():
        pattern = pattern_info["pattern"]
        
        for match in re.finditer(pattern, text):
            clauses.append({
                "clause_type": clause_type,
                "text": match.group(),
                "description": pattern_info["description"],
                "risk_level": pattern_info["risk_level"],
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.85
            })
    
    return clauses

def generate_summary(text: str) -> str:
    """Generate a summary of the document."""
    # Simple extractive summary
    sentences = text.split('.')
    
    # Take first few sentences as summary
    summary_sentences = sentences[:3]
    summary = '. '.join(s.strip() for s in summary_sentences if s.strip())
    
    if len(summary) > 500:
        summary = summary[:500] + "..."
    
    return summary or "Document summary not available."

def assess_risks(clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assess risks based on detected clauses."""
    risk_counts = {"high": 0, "medium": 0, "low": 0}
    
    for clause in clauses:
        risk_level = clause.get("risk_level", "low")
        risk_counts[risk_level] += 1
    
    total_clauses = len(clauses)
    
    if total_clauses == 0:
        overall_risk = "unknown"
    elif risk_counts["high"] > 0:
        overall_risk = "high"
    elif risk_counts["medium"] > 2:
        overall_risk = "medium"
    else:
        overall_risk = "low"
    
    return {
        "overall_risk": overall_risk,
        "risk_breakdown": risk_counts,
        "total_clauses": total_clauses,
        "recommendations": [
            "Review high-risk clauses carefully",
            "Consider legal consultation for complex terms",
            "Ensure compliance with applicable laws"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting Legal Intelligence Platform Demo Server...")
    print("📋 Available endpoints:")
    print("   - GET  /              : API information")
    print("   - GET  /health        : Health check")
    print("   - POST /api/v1/analyze: Analyze document")
    print("   - POST /api/v1/search-clauses: Search clauses")
    print("   - POST /api/v1/upload : Upload document")
    print("\n🌐 Server will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
