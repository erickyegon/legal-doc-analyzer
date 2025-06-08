"""
Regex Search Tool for the Legal Intelligence Platform.

This tool provides advanced regex pattern matching for detecting
legal clauses, terms, and document structures in legal documents.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Pattern
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

# LangChain imports
from langchain.tools import BaseTool

# Configure logging
logger = logging.getLogger(__name__)


class ClauseType(Enum):
    """Types of legal clauses for pattern matching."""
    TERMINATION = "termination"
    PAYMENT = "payment"
    LIABILITY = "liability"
    CONFIDENTIALITY = "confidentiality"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    FORCE_MAJEURE = "force_majeure"
    GOVERNING_LAW = "governing_law"
    DISPUTE_RESOLUTION = "dispute_resolution"
    INDEMNIFICATION = "indemnification"
    WARRANTY = "warranty"
    ASSIGNMENT = "assignment"
    AMENDMENT = "amendment"
    SEVERABILITY = "severability"
    ENTIRE_AGREEMENT = "entire_agreement"
    NOTICE = "notice"


@dataclass
class RegexMatch:
    """Data class for regex match results."""
    pattern_name: str
    clause_type: str
    matched_text: str
    start_position: int
    end_position: int
    confidence: float
    context: str
    groups: List[str]
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Result of regex search operation."""
    matches: List[RegexMatch]
    pattern_stats: Dict[str, int]
    total_matches: int
    confidence_score: float
    processing_time: float


class RegexSearchTool(BaseTool):
    """
    Advanced Regex Search Tool for legal clause detection.
    
    This tool provides:
    - Pre-built patterns for common legal clauses
    - Custom pattern compilation and testing
    - Context-aware matching with confidence scoring
    - Pattern performance analytics
    - Fuzzy matching capabilities
    - Multi-language pattern support
    """
    
    name = "regex_search"
    description = "Search for legal clause patterns using advanced regex matching"
    
    def __init__(self):
        """Initialize the Regex Search Tool."""
        super().__init__()
        
        # Initialize pattern library
        self.patterns = self._initialize_patterns()
        
        # Compiled patterns cache
        self.compiled_patterns: Dict[str, Pattern] = {}
        self._compile_patterns()
        
        # Pattern performance tracking
        self.pattern_stats = {}
    
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive legal clause patterns."""
        patterns = {}
        
        # Termination Clauses
        patterns["termination_basic"] = {
            "clause_type": ClauseType.TERMINATION,
            "pattern": r"(?i)\b(?:terminat|end|expir|cancel|dissolv)\w*\s+(?:this\s+)?(?:agreement|contract|lease)\b",
            "description": "Basic termination clause detection",
            "confidence": 0.8,
            "examples": ["terminate this agreement", "expiry of contract", "cancellation of lease"]
        }
        
        patterns["termination_notice"] = {
            "clause_type": ClauseType.TERMINATION,
            "pattern": r"(?i)(?:upon|after|with)\s+(\d+)\s+(days?|months?|years?)\s+(?:written\s+)?notice",
            "description": "Termination notice period detection",
            "confidence": 0.9,
            "examples": ["upon 30 days written notice", "after 60 days notice"]
        }
        
        patterns["termination_cause"] = {
            "clause_type": ClauseType.TERMINATION,
            "pattern": r"(?i)terminat\w*\s+(?:for\s+)?(?:cause|breach|default|material\s+breach)",
            "description": "Termination for cause clauses",
            "confidence": 0.85,
            "examples": ["termination for cause", "terminate for material breach"]
        }
        
        # Payment Clauses
        patterns["payment_terms"] = {
            "clause_type": ClauseType.PAYMENT,
            "pattern": r"(?i)payment\s+(?:shall\s+be\s+)?(?:due|made|payable)\s+(?:within\s+)?(\d+)\s+(days?|months?)",
            "description": "Payment terms and due dates",
            "confidence": 0.9,
            "examples": ["payment due within 30 days", "payment shall be made within 60 days"]
        }
        
        patterns["late_fees"] = {
            "clause_type": ClauseType.PAYMENT,
            "pattern": r"(?i)(?:late\s+fee|penalty|interest)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(%|percent|dollars?)",
            "description": "Late payment fees and penalties",
            "confidence": 0.85,
            "examples": ["late fee of 1.5%", "penalty of $50", "interest of 2 percent"]
        }
        
        patterns["payment_methods"] = {
            "clause_type": ClauseType.PAYMENT,
            "pattern": r"(?i)payment\s+(?:shall\s+be\s+made\s+)?by\s+(check|wire\s+transfer|ach|credit\s+card|cash)",
            "description": "Accepted payment methods",
            "confidence": 0.8,
            "examples": ["payment by wire transfer", "payment shall be made by check"]
        }
        
        # Liability Clauses
        patterns["liability_limitation"] = {
            "clause_type": ClauseType.LIABILITY,
            "pattern": r"(?i)(?:limit|cap|maximum)\s+(?:of\s+)?liability\s+(?:shall\s+)?(?:not\s+exceed|be\s+limited\s+to)\s+\$?([\d,]+)",
            "description": "Liability limitation amounts",
            "confidence": 0.9,
            "examples": ["limit of liability shall not exceed $100,000", "maximum liability of $50,000"]
        }
        
        patterns["liability_exclusion"] = {
            "clause_type": ClauseType.LIABILITY,
            "pattern": r"(?i)(?:exclud|disclaim|not\s+liable)\w*\s+(?:for\s+)?(?:any\s+)?(?:indirect|consequential|incidental|special|punitive)\s+damages",
            "description": "Liability exclusion clauses",
            "confidence": 0.85,
            "examples": ["exclude any consequential damages", "not liable for indirect damages"]
        }
        
        # Confidentiality Clauses
        patterns["confidentiality_basic"] = {
            "clause_type": ClauseType.CONFIDENTIALITY,
            "pattern": r"(?i)(?:confidential|proprietary|trade\s+secret)\s+information",
            "description": "Basic confidentiality terms",
            "confidence": 0.8,
            "examples": ["confidential information", "proprietary information", "trade secret information"]
        }
        
        patterns["confidentiality_duration"] = {
            "clause_type": ClauseType.CONFIDENTIALITY,
            "pattern": r"(?i)confidentiality\s+(?:obligations?\s+)?(?:shall\s+)?(?:survive|remain\s+in\s+effect)\s+for\s+(\d+)\s+(years?|months?)",
            "description": "Confidentiality duration clauses",
            "confidence": 0.9,
            "examples": ["confidentiality shall survive for 5 years", "confidentiality obligations remain in effect for 3 years"]
        }
        
        # Intellectual Property Clauses
        patterns["ip_ownership"] = {
            "clause_type": ClauseType.INTELLECTUAL_PROPERTY,
            "pattern": r"(?i)(?:intellectual\s+property|copyright|patent|trademark|trade\s+secret)\s+(?:rights?\s+)?(?:shall\s+)?(?:belong\s+to|be\s+owned\s+by|vest\s+in)",
            "description": "IP ownership clauses",
            "confidence": 0.85,
            "examples": ["intellectual property rights shall belong to", "copyright shall be owned by"]
        }
        
        patterns["ip_license"] = {
            "clause_type": ClauseType.INTELLECTUAL_PROPERTY,
            "pattern": r"(?i)(?:grant|license|permit)\s+(?:a\s+)?(?:non-exclusive|exclusive|limited|perpetual)?\s*(?:license|right)\s+to\s+use",
            "description": "IP licensing clauses",
            "confidence": 0.8,
            "examples": ["grant a non-exclusive license to use", "license the right to use"]
        }
        
        # Force Majeure Clauses
        patterns["force_majeure"] = {
            "clause_type": ClauseType.FORCE_MAJEURE,
            "pattern": r"(?i)(?:force\s+majeure|act\s+of\s+god|unforeseeable\s+circumstances|beyond\s+(?:the\s+)?reasonable\s+control)",
            "description": "Force majeure event definitions",
            "confidence": 0.9,
            "examples": ["force majeure", "act of god", "unforeseeable circumstances"]
        }
        
        patterns["force_majeure_events"] = {
            "clause_type": ClauseType.FORCE_MAJEURE,
            "pattern": r"(?i)(?:war|terrorism|earthquake|flood|fire|pandemic|epidemic|government\s+action|strike|labor\s+dispute)",
            "description": "Specific force majeure events",
            "confidence": 0.7,
            "examples": ["war", "pandemic", "government action", "labor dispute"]
        }
        
        # Governing Law Clauses
        patterns["governing_law"] = {
            "clause_type": ClauseType.GOVERNING_LAW,
            "pattern": r"(?i)(?:governed\s+by|subject\s+to|construed\s+in\s+accordance\s+with)\s+(?:the\s+)?laws?\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            "description": "Governing law clauses",
            "confidence": 0.9,
            "examples": ["governed by the laws of California", "subject to laws of New York"]
        }
        
        patterns["jurisdiction"] = {
            "clause_type": ClauseType.GOVERNING_LAW,
            "pattern": r"(?i)(?:exclusive\s+)?jurisdiction\s+(?:of\s+)?(?:the\s+)?courts?\s+(?:of\s+|in\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            "description": "Jurisdiction clauses",
            "confidence": 0.85,
            "examples": ["exclusive jurisdiction of the courts of Delaware", "jurisdiction in California"]
        }
        
        # Dispute Resolution Clauses
        patterns["arbitration"] = {
            "clause_type": ClauseType.DISPUTE_RESOLUTION,
            "pattern": r"(?i)(?:binding\s+)?arbitration\s+(?:in\s+accordance\s+with|under|pursuant\s+to)\s+(?:the\s+)?(?:rules\s+of\s+)?([A-Z][A-Z]+|American\s+Arbitration\s+Association)",
            "description": "Arbitration clauses",
            "confidence": 0.9,
            "examples": ["binding arbitration under AAA rules", "arbitration in accordance with JAMS"]
        }
        
        patterns["mediation"] = {
            "clause_type": ClauseType.DISPUTE_RESOLUTION,
            "pattern": r"(?i)(?:first\s+attempt\s+to\s+resolve|prior\s+to\s+litigation)\s+(?:through\s+)?mediation",
            "description": "Mediation requirements",
            "confidence": 0.8,
            "examples": ["first attempt to resolve through mediation", "prior to litigation mediation"]
        }
        
        # Indemnification Clauses
        patterns["indemnification"] = {
            "clause_type": ClauseType.INDEMNIFICATION,
            "pattern": r"(?i)(?:indemnify|hold\s+harmless|defend)\s+(?:and\s+)?(?:hold\s+harmless|defend|indemnify)\s+(?:against|from)",
            "description": "Indemnification obligations",
            "confidence": 0.85,
            "examples": ["indemnify and hold harmless against", "defend and indemnify from"]
        }
        
        # Warranty Clauses
        patterns["warranty_disclaimer"] = {
            "clause_type": ClauseType.WARRANTY,
            "pattern": r"(?i)(?:disclaim|exclude|without)\s+(?:all\s+)?(?:warranties|representations)\s+(?:express\s+or\s+implied|of\s+any\s+kind)",
            "description": "Warranty disclaimer clauses",
            "confidence": 0.85,
            "examples": ["disclaim all warranties express or implied", "without warranties of any kind"]
        }
        
        patterns["warranty_express"] = {
            "clause_type": ClauseType.WARRANTY,
            "pattern": r"(?i)(?:warrant|guarantee|represent)\s+(?:and\s+covenant\s+)?that",
            "description": "Express warranty clauses",
            "confidence": 0.8,
            "examples": ["warrant and covenant that", "guarantee that", "represent that"]
        }
        
        # Notice Clauses
        patterns["notice_method"] = {
            "clause_type": ClauseType.NOTICE,
            "pattern": r"(?i)notice\s+(?:shall\s+be\s+)?(?:given|provided|delivered)\s+(?:by\s+)?(?:written|email|certified\s+mail|registered\s+mail)",
            "description": "Notice delivery methods",
            "confidence": 0.8,
            "examples": ["notice shall be given by written", "notice provided by email"]
        }
        
        patterns["notice_address"] = {
            "clause_type": ClauseType.NOTICE,
            "pattern": r"(?i)notice\s+(?:shall\s+be\s+sent\s+)?to\s+(?:the\s+)?(?:address|email)\s+(?:set\s+forth|specified|listed)",
            "description": "Notice address requirements",
            "confidence": 0.75,
            "examples": ["notice to the address set forth", "notice shall be sent to the email specified"]
        }
        
        return patterns
    
    def _compile_patterns(self):
        """Compile all regex patterns for better performance."""
        for pattern_name, pattern_data in self.patterns.items():
            try:
                compiled_pattern = re.compile(pattern_data["pattern"], re.IGNORECASE | re.MULTILINE)
                self.compiled_patterns[pattern_name] = compiled_pattern
            except re.error as e:
                logger.error(f"Failed to compile pattern '{pattern_name}': {e}")
    
    def _run(self, text: str, **kwargs) -> str:
        """
        Run the regex search tool.
        
        Args:
            text (str): Text to search
            **kwargs: Additional arguments
            
        Returns:
            str: Search results summary
        """
        try:
            clause_types = kwargs.get('clause_types', None)
            min_confidence = kwargs.get('min_confidence', 0.5)
            
            result = self.search_patterns(
                text=text,
                clause_types=clause_types,
                min_confidence=min_confidence
            )
            
            # Format summary
            summary = f"Found {result.total_matches} clause matches:\n"
            
            for clause_type, count in result.pattern_stats.items():
                summary += f"- {clause_type}: {count}\n"
            
            summary += f"\nOverall Confidence: {result.confidence_score:.2f}"
            summary += f"\nProcessing Time: {result.processing_time:.3f}s"
            
            return summary
            
        except Exception as e:
            logger.error(f"Regex search failed: {e}")
            return f"Regex search failed: {str(e)}"
    
    def search_patterns(self,
                       text: str,
                       clause_types: Optional[List[str]] = None,
                       pattern_names: Optional[List[str]] = None,
                       min_confidence: float = 0.5,
                       context_length: int = 100) -> SearchResult:
        """
        Search for legal clause patterns in text.
        
        Args:
            text (str): Text to search
            clause_types (List[str], optional): Filter by clause types
            pattern_names (List[str], optional): Filter by specific patterns
            min_confidence (float): Minimum confidence threshold
            context_length (int): Length of context to capture
            
        Returns:
            SearchResult: Comprehensive search results
        """
        start_time = datetime.now()
        matches = []
        
        try:
            # Determine which patterns to search
            patterns_to_search = self._get_patterns_to_search(clause_types, pattern_names)
            
            for pattern_name in patterns_to_search:
                pattern_data = self.patterns[pattern_name]
                compiled_pattern = self.compiled_patterns.get(pattern_name)
                
                if not compiled_pattern:
                    continue
                
                # Search for matches
                pattern_matches = self._search_single_pattern(
                    text, pattern_name, pattern_data, compiled_pattern, context_length
                )
                
                # Filter by confidence
                filtered_matches = [
                    match for match in pattern_matches 
                    if match.confidence >= min_confidence
                ]
                
                matches.extend(filtered_matches)
                
                # Update pattern statistics
                self.pattern_stats[pattern_name] = self.pattern_stats.get(pattern_name, 0) + len(filtered_matches)
            
            # Calculate statistics
            pattern_stats = self._calculate_pattern_stats(matches)
            confidence_score = self._calculate_confidence_score(matches)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return SearchResult(
                matches=matches,
                pattern_stats=pattern_stats,
                total_matches=len(matches),
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Pattern search failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return SearchResult(
                matches=[],
                pattern_stats={},
                total_matches=0,
                confidence_score=0.0,
                processing_time=processing_time
            )
    
    def _get_patterns_to_search(self, 
                               clause_types: Optional[List[str]], 
                               pattern_names: Optional[List[str]]) -> List[str]:
        """Determine which patterns to search based on filters."""
        if pattern_names:
            return [name for name in pattern_names if name in self.patterns]
        
        if clause_types:
            patterns_to_search = []
            for pattern_name, pattern_data in self.patterns.items():
                if pattern_data["clause_type"].value in clause_types:
                    patterns_to_search.append(pattern_name)
            return patterns_to_search
        
        # Return all patterns if no filters
        return list(self.patterns.keys())
    
    def _search_single_pattern(self,
                              text: str,
                              pattern_name: str,
                              pattern_data: Dict[str, Any],
                              compiled_pattern: Pattern,
                              context_length: int) -> List[RegexMatch]:
        """Search for a single pattern in text."""
        matches = []
        
        try:
            for match in compiled_pattern.finditer(text):
                # Extract context
                start_pos = max(0, match.start() - context_length)
                end_pos = min(len(text), match.end() + context_length)
                context = text[start_pos:end_pos]
                
                # Extract groups
                groups = list(match.groups())
                
                # Calculate confidence (can be enhanced with ML models)
                confidence = self._calculate_match_confidence(
                    match, pattern_data, context
                )
                
                regex_match = RegexMatch(
                    pattern_name=pattern_name,
                    clause_type=pattern_data["clause_type"].value,
                    matched_text=match.group(0),
                    start_position=match.start(),
                    end_position=match.end(),
                    confidence=confidence,
                    context=context,
                    groups=groups,
                    metadata={
                        "pattern_description": pattern_data["description"],
                        "base_confidence": pattern_data["confidence"],
                        "match_span": match.span()
                    }
                )
                
                matches.append(regex_match)
                
        except Exception as e:
            logger.error(f"Single pattern search failed for {pattern_name}: {e}")
        
        return matches
    
    def _calculate_match_confidence(self,
                                   match: re.Match,
                                   pattern_data: Dict[str, Any],
                                   context: str) -> float:
        """Calculate confidence score for a regex match."""
        base_confidence = pattern_data.get("confidence", 0.5)
        
        # Adjust confidence based on various factors
        confidence_adjustments = 0.0
        
        # Length of match (longer matches generally more reliable)
        match_length = len(match.group(0))
        if match_length > 50:
            confidence_adjustments += 0.1
        elif match_length < 10:
            confidence_adjustments -= 0.1
        
        # Context quality (presence of legal keywords)
        legal_keywords = ["agreement", "contract", "party", "shall", "hereby", "whereas"]
        context_lower = context.lower()
        keyword_count = sum(1 for keyword in legal_keywords if keyword in context_lower)
        
        if keyword_count >= 3:
            confidence_adjustments += 0.1
        elif keyword_count == 0:
            confidence_adjustments -= 0.1
        
        # Sentence structure (complete sentences are better)
        if context.count('.') >= 1 and context.count(',') >= 1:
            confidence_adjustments += 0.05
        
        final_confidence = min(1.0, max(0.0, base_confidence + confidence_adjustments))
        return round(final_confidence, 2)
    
    def _calculate_pattern_stats(self, matches: List[RegexMatch]) -> Dict[str, int]:
        """Calculate statistics by clause type."""
        stats = {}
        for match in matches:
            clause_type = match.clause_type
            stats[clause_type] = stats.get(clause_type, 0) + 1
        return stats
    
    def _calculate_confidence_score(self, matches: List[RegexMatch]) -> float:
        """Calculate overall confidence score."""
        if not matches:
            return 0.0
        
        total_confidence = sum(match.confidence for match in matches)
        return round(total_confidence / len(matches), 2)
    
    def add_custom_pattern(self,
                          pattern_name: str,
                          pattern: str,
                          clause_type: ClauseType,
                          description: str,
                          confidence: float = 0.7) -> bool:
        """
        Add a custom regex pattern to the library.
        
        Args:
            pattern_name (str): Unique name for the pattern
            pattern (str): Regex pattern string
            clause_type (ClauseType): Type of clause this pattern detects
            description (str): Description of what the pattern matches
            confidence (float): Base confidence score for this pattern
            
        Returns:
            bool: Success status
        """
        try:
            # Test pattern compilation
            compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            
            # Add to patterns library
            self.patterns[pattern_name] = {
                "clause_type": clause_type,
                "pattern": pattern,
                "description": description,
                "confidence": confidence,
                "custom": True
            }
            
            # Add to compiled patterns
            self.compiled_patterns[pattern_name] = compiled_pattern
            
            logger.info(f"Added custom pattern: {pattern_name}")
            return True
            
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to add custom pattern: {e}")
            return False
    
    def test_pattern(self, pattern: str, test_text: str) -> Dict[str, Any]:
        """
        Test a regex pattern against sample text.
        
        Args:
            pattern (str): Regex pattern to test
            test_text (str): Text to test against
            
        Returns:
            Dict: Test results including matches and performance
        """
        try:
            start_time = datetime.now()
            
            # Compile pattern
            compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            
            # Find matches
            matches = []
            for match in compiled_pattern.finditer(test_text):
                matches.append({
                    "matched_text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "groups": list(match.groups())
                })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "pattern": pattern,
                "matches_found": len(matches),
                "matches": matches,
                "processing_time": processing_time,
                "success": True
            }
            
        except re.error as e:
            return {
                "pattern": pattern,
                "error": f"Invalid regex: {str(e)}",
                "success": False
            }
        except Exception as e:
            return {
                "pattern": pattern,
                "error": f"Test failed: {str(e)}",
                "success": False
            }
    
    def get_pattern_library(self) -> Dict[str, Any]:
        """
        Get information about all available patterns.
        
        Returns:
            Dict: Pattern library information
        """
        library_info = {
            "total_patterns": len(self.patterns),
            "clause_types": {},
            "patterns": {}
        }
        
        # Count patterns by clause type
        for pattern_name, pattern_data in self.patterns.items():
            clause_type = pattern_data["clause_type"].value
            library_info["clause_types"][clause_type] = library_info["clause_types"].get(clause_type, 0) + 1
            
            # Add pattern details
            library_info["patterns"][pattern_name] = {
                "clause_type": clause_type,
                "description": pattern_data["description"],
                "confidence": pattern_data["confidence"],
                "custom": pattern_data.get("custom", False)
            }
        
        return library_info
