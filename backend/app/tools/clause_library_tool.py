"""
Clause Library Vector Database Tool for the Legal Intelligence Platform.

This tool manages a comprehensive library of standard legal clauses with
vector similarity search capabilities for clause matching and analysis.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

# Vector database libraries
import chromadb
from chromadb.config import Settings
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# LangChain integration
from langchain.tools import BaseTool
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Text processing
from fuzzywuzzy import fuzz
import re

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class StandardClause:
    """Data class for standard legal clauses."""
    id: str
    title: str
    content: str
    category: str
    subcategory: Optional[str]
    jurisdiction: str
    clause_type: str  # 'boilerplate', 'negotiable', 'critical'
    risk_level: str  # 'low', 'medium', 'high'
    tags: List[str]
    source: str
    version: str
    created_date: datetime
    updated_date: datetime
    usage_count: int
    metadata: Dict[str, Any]


@dataclass
class ClauseMatch:
    """Data class for clause matching results."""
    matched_clause: StandardClause
    similarity_score: float
    match_type: str  # 'exact', 'semantic', 'fuzzy'
    confidence: float
    highlighted_differences: Optional[str]
    recommendations: List[str]


class ClauseLibraryTool(BaseTool):
    """
    Clause Library Vector Database Tool.
    
    This tool provides:
    - Storage and retrieval of standard legal clauses
    - Vector similarity search for clause matching
    - Fuzzy matching for clause variations
    - Clause categorization and tagging
    - Risk assessment for clauses
    - Usage analytics and recommendations
    """
    
    name = "clause_library"
    description = "Vector database tool for storing and searching standard legal clauses"
    
    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chroma_persist_directory: str = "./data/clause_library",
                 similarity_threshold: float = 0.7):
        """
        Initialize the Clause Library Tool.
        
        Args:
            embedding_model (str): Sentence transformer model for embeddings
            chroma_persist_directory (str): Directory for ChromaDB persistence
            similarity_threshold (float): Minimum similarity score for matches
        """
        super().__init__()
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
            self.embeddings = None
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.chroma_collection = None
        self.chroma_persist_directory = chroma_persist_directory
        self._initialize_chromadb()
        
        # Initialize FAISS index
        self.faiss_index = None
        self.clause_metadata = {}
        self._initialize_faiss()
        
        # Configuration
        self.similarity_threshold = similarity_threshold
        
        # Clause categories
        self.clause_categories = {
            'termination': ['termination', 'end', 'expiry', 'cancellation'],
            'payment': ['payment', 'fee', 'cost', 'invoice', 'billing'],
            'liability': ['liability', 'damages', 'indemnification', 'limitation'],
            'confidentiality': ['confidential', 'non-disclosure', 'proprietary', 'trade secret'],
            'intellectual_property': ['intellectual property', 'copyright', 'patent', 'trademark'],
            'force_majeure': ['force majeure', 'act of god', 'unforeseeable circumstances'],
            'governing_law': ['governing law', 'jurisdiction', 'applicable law', 'venue'],
            'dispute_resolution': ['dispute', 'arbitration', 'mediation', 'litigation'],
            'warranty': ['warranty', 'guarantee', 'representation', 'assurance'],
            'assignment': ['assignment', 'transfer', 'delegation', 'succession']
        }
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB for clause storage."""
        try:
            # Create ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="legal_clauses",
                metadata={"description": "Standard legal clauses library"}
            )
            
            logger.info(f"Initialized ChromaDB at {self.chroma_persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
            self.chroma_collection = None
    
    def _initialize_faiss(self):
        """Initialize FAISS index for fast similarity search."""
        try:
            # Create FAISS index (will be populated when clauses are added)
            embedding_dim = 384  # Default for all-MiniLM-L6-v2
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
            
            logger.info("Initialized FAISS index")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            self.faiss_index = None
    
    def _run(self, query: str, **kwargs) -> str:
        """
        Run the clause library tool.
        
        Args:
            query (str): Search query for clauses
            **kwargs: Additional arguments
            
        Returns:
            str: Search results summary
        """
        try:
            matches = self.search_clauses(query, **kwargs)
            
            if matches:
                summary = f"Found {len(matches)} clause matches:\n"
                for i, match in enumerate(matches[:3], 1):
                    summary += f"{i}. {match.matched_clause.title} (similarity: {match.similarity_score:.2f})\n"
                    summary += f"   Category: {match.matched_clause.category}\n"
                    summary += f"   Risk Level: {match.matched_clause.risk_level}\n\n"
            else:
                summary = "No matching clauses found."
            
            return summary
            
        except Exception as e:
            logger.error(f"Clause search failed: {e}")
            return f"Clause search failed: {str(e)}"
    
    def add_clause(self, clause: StandardClause) -> bool:
        """
        Add a standard clause to the library.
        
        Args:
            clause (StandardClause): Clause to add
            
        Returns:
            bool: Success status
        """
        try:
            # Generate embedding
            if self.embedding_model:
                embedding = self.embedding_model.encode([clause.content])[0]
            else:
                logger.warning("No embedding model available")
                return False
            
            # Add to ChromaDB
            if self.chroma_collection:
                self.chroma_collection.add(
                    documents=[clause.content],
                    embeddings=[embedding.tolist()],
                    metadatas=[{
                        'id': clause.id,
                        'title': clause.title,
                        'category': clause.category,
                        'subcategory': clause.subcategory or '',
                        'jurisdiction': clause.jurisdiction,
                        'clause_type': clause.clause_type,
                        'risk_level': clause.risk_level,
                        'tags': json.dumps(clause.tags),
                        'source': clause.source,
                        'version': clause.version,
                        'created_date': clause.created_date.isoformat(),
                        'updated_date': clause.updated_date.isoformat(),
                        'usage_count': clause.usage_count
                    }],
                    ids=[clause.id]
                )
            
            # Add to FAISS index
            if self.faiss_index:
                # Normalize embedding for cosine similarity
                normalized_embedding = embedding / np.linalg.norm(embedding)
                self.faiss_index.add(normalized_embedding.reshape(1, -1))
                self.clause_metadata[self.faiss_index.ntotal - 1] = clause
            
            logger.info(f"Added clause: {clause.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add clause: {e}")
            return False
    
    def search_clauses(self,
                      query: str,
                      category: Optional[str] = None,
                      risk_level: Optional[str] = None,
                      jurisdiction: Optional[str] = None,
                      top_k: int = 10,
                      include_fuzzy: bool = True) -> List[ClauseMatch]:
        """
        Search for clauses using vector similarity and fuzzy matching.
        
        Args:
            query (str): Search query
            category (str, optional): Filter by category
            risk_level (str, optional): Filter by risk level
            jurisdiction (str, optional): Filter by jurisdiction
            top_k (int): Number of results to return
            include_fuzzy (bool): Include fuzzy matching results
            
        Returns:
            List[ClauseMatch]: Matching clauses with similarity scores
        """
        matches = []
        
        try:
            # Vector similarity search
            vector_matches = self._vector_search(query, category, risk_level, jurisdiction, top_k)
            matches.extend(vector_matches)
            
            # Fuzzy matching search
            if include_fuzzy:
                fuzzy_matches = self._fuzzy_search(query, category, risk_level, jurisdiction, top_k)
                matches.extend(fuzzy_matches)
            
            # Remove duplicates and sort by similarity
            unique_matches = {}
            for match in matches:
                clause_id = match.matched_clause.id
                if clause_id not in unique_matches or match.similarity_score > unique_matches[clause_id].similarity_score:
                    unique_matches[clause_id] = match
            
            # Sort by similarity score
            sorted_matches = sorted(unique_matches.values(), key=lambda x: x.similarity_score, reverse=True)
            
            return sorted_matches[:top_k]
            
        except Exception as e:
            logger.error(f"Clause search failed: {e}")
            return []
    
    def _vector_search(self,
                      query: str,
                      category: Optional[str] = None,
                      risk_level: Optional[str] = None,
                      jurisdiction: Optional[str] = None,
                      top_k: int = 10) -> List[ClauseMatch]:
        """Perform vector similarity search using ChromaDB."""
        matches = []
        
        try:
            if not self.chroma_collection or not self.embedding_model:
                return matches
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Build filter conditions
            where_conditions = {}
            if category:
                where_conditions['category'] = category
            if risk_level:
                where_conditions['risk_level'] = risk_level
            if jurisdiction:
                where_conditions['jurisdiction'] = jurisdiction
            
            # Search in ChromaDB
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_conditions if where_conditions else None
            )
            
            # Process results
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Convert distance to similarity score
                similarity_score = 1 - distance
                
                if similarity_score >= self.similarity_threshold:
                    # Create StandardClause object
                    clause = StandardClause(
                        id=metadata['id'],
                        title=metadata['title'],
                        content=doc,
                        category=metadata['category'],
                        subcategory=metadata.get('subcategory'),
                        jurisdiction=metadata['jurisdiction'],
                        clause_type=metadata['clause_type'],
                        risk_level=metadata['risk_level'],
                        tags=json.loads(metadata.get('tags', '[]')),
                        source=metadata['source'],
                        version=metadata['version'],
                        created_date=datetime.fromisoformat(metadata['created_date']),
                        updated_date=datetime.fromisoformat(metadata['updated_date']),
                        usage_count=metadata['usage_count'],
                        metadata={}
                    )
                    
                    # Create match object
                    match = ClauseMatch(
                        matched_clause=clause,
                        similarity_score=similarity_score,
                        match_type='semantic',
                        confidence=similarity_score,
                        highlighted_differences=None,
                        recommendations=self._generate_recommendations(clause, query)
                    )
                    
                    matches.append(match)
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        
        return matches
    
    def _fuzzy_search(self,
                     query: str,
                     category: Optional[str] = None,
                     risk_level: Optional[str] = None,
                     jurisdiction: Optional[str] = None,
                     top_k: int = 10) -> List[ClauseMatch]:
        """Perform fuzzy string matching search."""
        matches = []
        
        try:
            if not self.chroma_collection:
                return matches
            
            # Get all clauses for fuzzy matching
            all_results = self.chroma_collection.get()
            
            for doc, metadata in zip(all_results['documents'], all_results['metadatas']):
                # Apply filters
                if category and metadata['category'] != category:
                    continue
                if risk_level and metadata['risk_level'] != risk_level:
                    continue
                if jurisdiction and metadata['jurisdiction'] != jurisdiction:
                    continue
                
                # Calculate fuzzy similarity
                title_similarity = fuzz.partial_ratio(query.lower(), metadata['title'].lower()) / 100.0
                content_similarity = fuzz.partial_ratio(query.lower(), doc.lower()) / 100.0
                
                # Use the higher similarity score
                fuzzy_score = max(title_similarity, content_similarity)
                
                if fuzzy_score >= self.similarity_threshold:
                    # Create StandardClause object
                    clause = StandardClause(
                        id=metadata['id'],
                        title=metadata['title'],
                        content=doc,
                        category=metadata['category'],
                        subcategory=metadata.get('subcategory'),
                        jurisdiction=metadata['jurisdiction'],
                        clause_type=metadata['clause_type'],
                        risk_level=metadata['risk_level'],
                        tags=json.loads(metadata.get('tags', '[]')),
                        source=metadata['source'],
                        version=metadata['version'],
                        created_date=datetime.fromisoformat(metadata['created_date']),
                        updated_date=datetime.fromisoformat(metadata['updated_date']),
                        usage_count=metadata['usage_count'],
                        metadata={}
                    )
                    
                    # Create match object
                    match = ClauseMatch(
                        matched_clause=clause,
                        similarity_score=fuzzy_score,
                        match_type='fuzzy',
                        confidence=fuzzy_score * 0.8,  # Lower confidence for fuzzy matches
                        highlighted_differences=self._highlight_differences(query, doc),
                        recommendations=self._generate_recommendations(clause, query)
                    )
                    
                    matches.append(match)
            
        except Exception as e:
            logger.error(f"Fuzzy search failed: {e}")
        
        return matches
    
    def _generate_recommendations(self, clause: StandardClause, query: str) -> List[str]:
        """Generate recommendations for clause usage."""
        recommendations = []
        
        # Risk-based recommendations
        if clause.risk_level == 'high':
            recommendations.append("⚠️ High-risk clause - requires careful legal review")
        elif clause.risk_level == 'medium':
            recommendations.append("⚡ Medium-risk clause - consider legal consultation")
        
        # Category-specific recommendations
        if clause.category == 'liability':
            recommendations.append("💼 Consider liability caps and exclusions")
        elif clause.category == 'termination':
            recommendations.append("📅 Review termination notice periods and conditions")
        elif clause.category == 'payment':
            recommendations.append("💰 Verify payment terms and late fee provisions")
        
        # Usage-based recommendations
        if clause.usage_count > 100:
            recommendations.append("✅ Frequently used clause - well-tested")
        elif clause.usage_count < 10:
            recommendations.append("🔍 Rarely used clause - consider alternatives")
        
        return recommendations
    
    def _highlight_differences(self, query: str, content: str) -> str:
        """Highlight differences between query and matched content."""
        # Simple implementation - could be enhanced with more sophisticated diff algorithms
        query_words = set(query.lower().split())
        content_words = content.lower().split()
        
        highlighted = []
        for word in content_words:
            if word in query_words:
                highlighted.append(f"**{word}**")
            else:
                highlighted.append(word)
        
        return " ".join(highlighted)
    
    def get_clause_by_id(self, clause_id: str) -> Optional[StandardClause]:
        """
        Retrieve a specific clause by ID.
        
        Args:
            clause_id (str): Clause identifier
            
        Returns:
            StandardClause: The requested clause or None
        """
        try:
            if not self.chroma_collection:
                return None
            
            results = self.chroma_collection.get(ids=[clause_id])
            
            if results['documents']:
                doc = results['documents'][0]
                metadata = results['metadatas'][0]
                
                return StandardClause(
                    id=metadata['id'],
                    title=metadata['title'],
                    content=doc,
                    category=metadata['category'],
                    subcategory=metadata.get('subcategory'),
                    jurisdiction=metadata['jurisdiction'],
                    clause_type=metadata['clause_type'],
                    risk_level=metadata['risk_level'],
                    tags=json.loads(metadata.get('tags', '[]')),
                    source=metadata['source'],
                    version=metadata['version'],
                    created_date=datetime.fromisoformat(metadata['created_date']),
                    updated_date=datetime.fromisoformat(metadata['updated_date']),
                    usage_count=metadata['usage_count'],
                    metadata={}
                )
            
        except Exception as e:
            logger.error(f"Failed to retrieve clause {clause_id}: {e}")
        
        return None
    
    def update_clause_usage(self, clause_id: str) -> bool:
        """
        Update usage count for a clause.
        
        Args:
            clause_id (str): Clause identifier
            
        Returns:
            bool: Success status
        """
        try:
            clause = self.get_clause_by_id(clause_id)
            if clause:
                clause.usage_count += 1
                clause.updated_date = datetime.now()
                
                # Update in ChromaDB
                if self.chroma_collection:
                    self.chroma_collection.update(
                        ids=[clause_id],
                        metadatas=[{
                            'usage_count': clause.usage_count,
                            'updated_date': clause.updated_date.isoformat()
                        }]
                    )
                
                return True
            
        except Exception as e:
            logger.error(f"Failed to update clause usage: {e}")
        
        return False
    
    def get_clause_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the clause library.
        
        Returns:
            Dict: Library statistics
        """
        try:
            if not self.chroma_collection:
                return {}
            
            # Get all clauses
            all_results = self.chroma_collection.get()
            
            if not all_results['metadatas']:
                return {'total_clauses': 0}
            
            # Calculate statistics
            total_clauses = len(all_results['metadatas'])
            categories = {}
            risk_levels = {}
            jurisdictions = {}
            
            for metadata in all_results['metadatas']:
                # Category distribution
                category = metadata['category']
                categories[category] = categories.get(category, 0) + 1
                
                # Risk level distribution
                risk_level = metadata['risk_level']
                risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
                
                # Jurisdiction distribution
                jurisdiction = metadata['jurisdiction']
                jurisdictions[jurisdiction] = jurisdictions.get(jurisdiction, 0) + 1
            
            return {
                'total_clauses': total_clauses,
                'categories': categories,
                'risk_levels': risk_levels,
                'jurisdictions': jurisdictions,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
