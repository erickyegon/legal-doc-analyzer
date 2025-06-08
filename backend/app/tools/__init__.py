"""
Advanced Tools package for the Legal Intelligence Platform.

This package contains specialized tools for legal document processing including:
- PDF Parser Tool with pdfminer/smart-pdf/DeepLake
- Clause Library Vector DB with similarity search
- Summarizer Tool with abstractive LLM-based summaries
- NER Tool with custom spaCy models and OpenAI function calls
- Regex Search Tool for clause pattern detection
- External API Tool for legal search APIs (Westlaw, LexisNexis)

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

from .pdf_parser_tool import PDFParserTool
from .clause_library_tool import ClauseLibraryTool
from .summarizer_tool import SummarizerTool
from .ner_tool import NERTool
from .regex_search_tool import RegexSearchTool
from .external_api_tool import ExternalAPITool

__all__ = [
    "PDFParserTool",
    "ClauseLibraryTool", 
    "SummarizerTool",
    "NERTool",
    "RegexSearchTool",
    "ExternalAPITool"
]
