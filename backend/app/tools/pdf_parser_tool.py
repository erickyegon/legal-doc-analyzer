"""
Advanced PDF Parser Tool for the Legal Intelligence Platform.

This tool provides comprehensive PDF parsing capabilities using pdfminer,
smart-pdf, and DeepLake for advanced chunking and document processing.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
from pathlib import Path

# PDF parsing libraries
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextContainer, LTTextBox, LTTextLine, LTChar, LTFigure, LTImage
import deeplake
from sentence_transformers import SentenceTransformer

# LangChain text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.tools import BaseTool
from langchain.schema import Document
import tiktoken

# Additional libraries
import numpy as np
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PDFChunk:
    """Data class for PDF chunks with metadata."""
    content: str
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int
    chunk_type: str  # 'text', 'table', 'header', 'footer', 'figure'
    bbox: Optional[Tuple[float, float, float, float]]  # (x0, y0, x1, y1)
    font_info: Optional[Dict[str, Any]]
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]


@dataclass
class PDFParseResult:
    """Result of PDF parsing operation."""
    document_id: str
    file_path: str
    total_pages: int
    total_chunks: int
    chunks: List[PDFChunk]
    document_metadata: Dict[str, Any]
    parsing_stats: Dict[str, Any]


class PDFParserTool(BaseTool):
    """
    Advanced PDF Parser Tool with chunking and embedding capabilities.
    
    This tool provides:
    - Layout-aware PDF parsing using pdfminer
    - Intelligent chunking with multiple strategies
    - DeepLake integration for vector storage
    - Semantic embeddings for chunks
    - Metadata extraction and preservation
    """
    
    name = "pdf_parser"
    description = "Advanced PDF parsing tool with layout-aware chunking and embedding generation"
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 deeplake_path: Optional[str] = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the PDF Parser Tool.
        
        Args:
            embedding_model (str): Sentence transformer model for embeddings
            deeplake_path (str): Path to DeepLake dataset
            chunk_size (int): Default chunk size for text splitting
            chunk_overlap (int): Overlap between chunks
        """
        super().__init__()
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
        # Initialize DeepLake dataset
        self.deeplake_path = deeplake_path or "./data/pdf_chunks"
        self.dataset = None
        self._initialize_deeplake()
        
        # Text splitting configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitters
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize token-based splitter
        try:
            self.token_splitter = TokenTextSplitter(
                chunk_size=chunk_size // 4,  # Approximate token count
                chunk_overlap=chunk_overlap // 4
            )
        except Exception as e:
            logger.warning(f"Failed to initialize token splitter: {e}")
            self.token_splitter = None
    
    def _initialize_deeplake(self):
        """Initialize DeepLake dataset for storing chunks."""
        try:
            # Create or load DeepLake dataset
            if os.path.exists(self.deeplake_path):
                self.dataset = deeplake.load(self.deeplake_path)
                logger.info(f"Loaded existing DeepLake dataset from {self.deeplake_path}")
            else:
                self.dataset = deeplake.empty(self.deeplake_path)
                
                # Create tensors for PDF chunks
                self.dataset.create_tensor('text', htype='text')
                self.dataset.create_tensor('embedding', htype='embedding')
                self.dataset.create_tensor('metadata', htype='json')
                self.dataset.create_tensor('page_number', htype='generic')
                self.dataset.create_tensor('chunk_index', htype='generic')
                self.dataset.create_tensor('document_id', htype='text')
                
                logger.info(f"Created new DeepLake dataset at {self.deeplake_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize DeepLake: {e}")
            self.dataset = None
    
    def _run(self, file_path: str, **kwargs) -> str:
        """
        Run the PDF parser tool.
        
        Args:
            file_path (str): Path to the PDF file
            **kwargs: Additional arguments
            
        Returns:
            str: Summary of parsing results
        """
        try:
            result = self.parse_pdf(file_path, **kwargs)
            
            summary = f"""PDF Parsing Complete:
- Document ID: {result.document_id}
- Total Pages: {result.total_pages}
- Total Chunks: {result.total_chunks}
- Chunk Types: {', '.join(set(chunk.chunk_type for chunk in result.chunks))}
- Processing Time: {result.parsing_stats.get('processing_time', 'N/A')} seconds
"""
            return summary
            
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            return f"PDF parsing failed: {str(e)}"
    
    def parse_pdf(self, 
                  file_path: str,
                  chunking_strategy: str = "hybrid",
                  preserve_layout: bool = True,
                  extract_metadata: bool = True,
                  generate_embeddings: bool = True) -> PDFParseResult:
        """
        Parse PDF with advanced chunking and metadata extraction.
        
        Args:
            file_path (str): Path to the PDF file
            chunking_strategy (str): Strategy for chunking ('semantic', 'fixed', 'hybrid')
            preserve_layout (bool): Whether to preserve layout information
            extract_metadata (bool): Whether to extract document metadata
            generate_embeddings (bool): Whether to generate embeddings for chunks
            
        Returns:
            PDFParseResult: Comprehensive parsing results
        """
        start_time = datetime.now()
        
        try:
            # Generate document ID
            document_id = self._generate_document_id(file_path)
            
            # Extract text and layout information
            if preserve_layout:
                chunks = self._extract_with_layout(file_path)
            else:
                chunks = self._extract_simple_text(file_path)
            
            # Apply chunking strategy
            if chunking_strategy == "semantic":
                chunks = self._apply_semantic_chunking(chunks)
            elif chunking_strategy == "fixed":
                chunks = self._apply_fixed_chunking(chunks)
            elif chunking_strategy == "hybrid":
                chunks = self._apply_hybrid_chunking(chunks)
            
            # Generate embeddings if requested
            if generate_embeddings and self.embedding_model:
                chunks = self._generate_embeddings(chunks)
            
            # Extract document metadata
            document_metadata = {}
            if extract_metadata:
                document_metadata = self._extract_document_metadata(file_path)
            
            # Calculate parsing statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            parsing_stats = {
                'processing_time': processing_time,
                'chunking_strategy': chunking_strategy,
                'preserve_layout': preserve_layout,
                'total_characters': sum(len(chunk.content) for chunk in chunks),
                'average_chunk_size': np.mean([len(chunk.content) for chunk in chunks]) if chunks else 0
            }
            
            # Store in DeepLake if available
            if self.dataset is not None:
                self._store_in_deeplake(document_id, chunks)
            
            # Create result
            result = PDFParseResult(
                document_id=document_id,
                file_path=file_path,
                total_pages=max((chunk.page_number for chunk in chunks), default=0),
                total_chunks=len(chunks),
                chunks=chunks,
                document_metadata=document_metadata,
                parsing_stats=parsing_stats
            )
            
            logger.info(f"Successfully parsed PDF: {file_path} ({len(chunks)} chunks)")
            return result
            
        except Exception as e:
            logger.error(f"PDF parsing failed for {file_path}: {e}")
            raise
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID based on file path and content."""
        file_stat = os.stat(file_path)
        content_hash = hashlib.md5(f"{file_path}_{file_stat.st_mtime}_{file_stat.st_size}".encode()).hexdigest()
        return f"doc_{content_hash[:16]}"
    
    def _extract_with_layout(self, file_path: str) -> List[PDFChunk]:
        """Extract text with layout preservation using pdfminer."""
        chunks = []
        chunk_index = 0
        
        try:
            for page_num, page_layout in enumerate(extract_pages(file_path), 1):
                page_chunks = self._process_page_layout(page_layout, page_num, chunk_index)
                chunks.extend(page_chunks)
                chunk_index += len(page_chunks)
                
        except Exception as e:
            logger.error(f"Layout extraction failed: {e}")
            # Fallback to simple text extraction
            return self._extract_simple_text(file_path)
        
        return chunks
    
    def _process_page_layout(self, page_layout, page_num: int, start_chunk_index: int) -> List[PDFChunk]:
        """Process individual page layout elements."""
        chunks = []
        chunk_index = start_chunk_index
        
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                # Extract text content
                text = element.get_text().strip()
                
                if len(text) > 20:  # Skip very short text elements
                    # Determine chunk type based on position and formatting
                    chunk_type = self._classify_text_element(element, page_layout)
                    
                    # Extract font information
                    font_info = self._extract_font_info(element)
                    
                    # Create chunk
                    chunk = PDFChunk(
                        content=text,
                        page_number=page_num,
                        chunk_index=chunk_index,
                        start_char=0,  # Will be updated during chunking
                        end_char=len(text),
                        chunk_type=chunk_type,
                        bbox=(element.bbox[0], element.bbox[1], element.bbox[2], element.bbox[3]),
                        font_info=font_info,
                        embedding=None,
                        metadata={
                            'element_type': type(element).__name__,
                            'bbox': element.bbox,
                            'page_number': page_num
                        }
                    )
                    
                    chunks.append(chunk)
                    chunk_index += 1
            
            elif isinstance(element, LTFigure):
                # Handle figures/images
                chunk = PDFChunk(
                    content=f"[FIGURE: {type(element).__name__}]",
                    page_number=page_num,
                    chunk_index=chunk_index,
                    start_char=0,
                    end_char=0,
                    chunk_type='figure',
                    bbox=(element.bbox[0], element.bbox[1], element.bbox[2], element.bbox[3]),
                    font_info=None,
                    embedding=None,
                    metadata={
                        'element_type': type(element).__name__,
                        'bbox': element.bbox,
                        'page_number': page_num
                    }
                )
                
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _classify_text_element(self, element, page_layout) -> str:
        """Classify text element type based on position and formatting."""
        # Simple heuristics for classification
        page_height = page_layout.height
        y_position = element.bbox[1]
        
        # Header detection (top 10% of page)
        if y_position > page_height * 0.9:
            return 'header'
        
        # Footer detection (bottom 10% of page)
        elif y_position < page_height * 0.1:
            return 'footer'
        
        # Check for table-like structure
        elif self._is_table_like(element):
            return 'table'
        
        else:
            return 'text'
    
    def _is_table_like(self, element) -> bool:
        """Detect if text element is part of a table."""
        text = element.get_text()
        
        # Simple heuristics for table detection
        lines = text.split('\n')
        if len(lines) > 1:
            # Check for consistent spacing or tab characters
            tab_count = sum(line.count('\t') for line in lines)
            if tab_count > len(lines) * 0.5:
                return True
            
            # Check for numeric patterns
            numeric_lines = sum(1 for line in lines if any(char.isdigit() for char in line))
            if numeric_lines > len(lines) * 0.7:
                return True
        
        return False
    
    def _extract_font_info(self, element) -> Dict[str, Any]:
        """Extract font information from text element."""
        font_info = {
            'fonts': [],
            'sizes': [],
            'colors': []
        }
        
        try:
            for text_line in element:
                if isinstance(text_line, LTTextLine):
                    for char in text_line:
                        if isinstance(char, LTChar):
                            font_info['fonts'].append(char.fontname)
                            font_info['sizes'].append(char.height)
                            # Note: color extraction would require more complex processing
        except Exception as e:
            logger.warning(f"Font extraction failed: {e}")
        
        # Get unique values
        font_info['fonts'] = list(set(font_info['fonts']))
        font_info['sizes'] = list(set(font_info['sizes']))
        
        return font_info
    
    def _extract_simple_text(self, file_path: str) -> List[PDFChunk]:
        """Fallback simple text extraction."""
        try:
            text = extract_text(file_path)
            
            # Create a single chunk for the entire document
            chunk = PDFChunk(
                content=text,
                page_number=1,
                chunk_index=0,
                start_char=0,
                end_char=len(text),
                chunk_type='text',
                bbox=None,
                font_info=None,
                embedding=None,
                metadata={'extraction_method': 'simple'}
            )
            
            return [chunk]
            
        except Exception as e:
            logger.error(f"Simple text extraction failed: {e}")
            return []

    def _apply_semantic_chunking(self, chunks: List[PDFChunk]) -> List[PDFChunk]:
        """Apply semantic chunking based on content similarity."""
        if not self.embedding_model:
            logger.warning("No embedding model available, falling back to fixed chunking")
            return self._apply_fixed_chunking(chunks)

        # Combine text chunks for semantic analysis
        combined_text = "\n\n".join(chunk.content for chunk in chunks if chunk.chunk_type == 'text')

        if not combined_text:
            return chunks

        # Use recursive text splitter with semantic awareness
        documents = self.recursive_splitter.split_text(combined_text)

        # Create new chunks with semantic boundaries
        semantic_chunks = []
        for i, doc_text in enumerate(documents):
            # Find the original chunk this text came from
            original_chunk = self._find_original_chunk(doc_text, chunks)

            chunk = PDFChunk(
                content=doc_text,
                page_number=original_chunk.page_number if original_chunk else 1,
                chunk_index=i,
                start_char=0,
                end_char=len(doc_text),
                chunk_type='text',
                bbox=original_chunk.bbox if original_chunk else None,
                font_info=original_chunk.font_info if original_chunk else None,
                embedding=None,
                metadata={
                    'chunking_method': 'semantic',
                    'original_chunk_id': original_chunk.chunk_index if original_chunk else None
                }
            )
            semantic_chunks.append(chunk)

        # Add non-text chunks (tables, figures, etc.)
        non_text_chunks = [chunk for chunk in chunks if chunk.chunk_type != 'text']
        semantic_chunks.extend(non_text_chunks)

        return semantic_chunks

    def _apply_fixed_chunking(self, chunks: List[PDFChunk]) -> List[PDFChunk]:
        """Apply fixed-size chunking."""
        fixed_chunks = []
        chunk_index = 0

        for chunk in chunks:
            if chunk.chunk_type == 'text' and len(chunk.content) > self.chunk_size:
                # Split large text chunks
                text_parts = self.recursive_splitter.split_text(chunk.content)

                for i, part in enumerate(text_parts):
                    new_chunk = PDFChunk(
                        content=part,
                        page_number=chunk.page_number,
                        chunk_index=chunk_index,
                        start_char=0,
                        end_char=len(part),
                        chunk_type=chunk.chunk_type,
                        bbox=chunk.bbox,
                        font_info=chunk.font_info,
                        embedding=None,
                        metadata={
                            'chunking_method': 'fixed',
                            'original_chunk_id': chunk.chunk_index,
                            'sub_chunk_index': i
                        }
                    )
                    fixed_chunks.append(new_chunk)
                    chunk_index += 1
            else:
                # Keep chunk as is
                chunk.chunk_index = chunk_index
                chunk.metadata['chunking_method'] = 'fixed'
                fixed_chunks.append(chunk)
                chunk_index += 1

        return fixed_chunks

    def _apply_hybrid_chunking(self, chunks: List[PDFChunk]) -> List[PDFChunk]:
        """Apply hybrid chunking combining semantic and fixed approaches."""
        # First apply semantic chunking
        semantic_chunks = self._apply_semantic_chunking(chunks)

        # Then apply size limits
        hybrid_chunks = []
        chunk_index = 0

        for chunk in semantic_chunks:
            if len(chunk.content) > self.chunk_size * 1.5:  # Allow some flexibility
                # Further split large chunks
                text_parts = self.recursive_splitter.split_text(chunk.content)

                for i, part in enumerate(text_parts):
                    new_chunk = PDFChunk(
                        content=part,
                        page_number=chunk.page_number,
                        chunk_index=chunk_index,
                        start_char=0,
                        end_char=len(part),
                        chunk_type=chunk.chunk_type,
                        bbox=chunk.bbox,
                        font_info=chunk.font_info,
                        embedding=None,
                        metadata={
                            'chunking_method': 'hybrid',
                            'original_chunk_id': chunk.chunk_index,
                            'sub_chunk_index': i
                        }
                    )
                    hybrid_chunks.append(new_chunk)
                    chunk_index += 1
            else:
                chunk.chunk_index = chunk_index
                chunk.metadata['chunking_method'] = 'hybrid'
                hybrid_chunks.append(chunk)
                chunk_index += 1

        return hybrid_chunks

    def _find_original_chunk(self, text: str, chunks: List[PDFChunk]) -> Optional[PDFChunk]:
        """Find the original chunk that contains the given text."""
        for chunk in chunks:
            if text in chunk.content:
                return chunk
        return None

    def _generate_embeddings(self, chunks: List[PDFChunk]) -> List[PDFChunk]:
        """Generate embeddings for text chunks."""
        if not self.embedding_model:
            logger.warning("No embedding model available")
            return chunks

        try:
            # Extract text content for embedding
            texts = [chunk.content for chunk in chunks if chunk.chunk_type in ['text', 'table']]

            if texts:
                # Generate embeddings in batches
                embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

                # Assign embeddings to chunks
                embedding_index = 0
                for chunk in chunks:
                    if chunk.chunk_type in ['text', 'table']:
                        chunk.embedding = embeddings[embedding_index].tolist()
                        embedding_index += 1

            logger.info(f"Generated embeddings for {len(texts)} chunks")

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")

        return chunks

    def _extract_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract document-level metadata."""
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'extraction_timestamp': datetime.now().isoformat()
        }

        try:
            # Additional metadata extraction could be added here
            # For example, using PyPDF2 or other libraries to extract PDF metadata
            pass

        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")

        return metadata

    def _store_in_deeplake(self, document_id: str, chunks: List[PDFChunk]):
        """Store chunks in DeepLake dataset."""
        if not self.dataset:
            logger.warning("DeepLake dataset not available")
            return

        try:
            for chunk in chunks:
                # Prepare data for storage
                embedding = chunk.embedding if chunk.embedding else [0.0] * 384  # Default embedding size

                # Append to dataset
                self.dataset.text.append(chunk.content)
                self.dataset.embedding.append(embedding)
                self.dataset.metadata.append(chunk.metadata)
                self.dataset.page_number.append(chunk.page_number)
                self.dataset.chunk_index.append(chunk.chunk_index)
                self.dataset.document_id.append(document_id)

            # Commit changes
            self.dataset.commit(f"Added {len(chunks)} chunks for document {document_id}")
            logger.info(f"Stored {len(chunks)} chunks in DeepLake")

        except Exception as e:
            logger.error(f"DeepLake storage failed: {e}")

    def search_similar_chunks(self,
                            query: str,
                            document_id: Optional[str] = None,
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query (str): Search query
            document_id (str, optional): Limit search to specific document
            top_k (int): Number of results to return

        Returns:
            List[Dict]: Similar chunks with similarity scores
        """
        if not self.dataset or not self.embedding_model:
            logger.warning("DeepLake dataset or embedding model not available")
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]

            # Search in DeepLake
            if document_id:
                # Filter by document ID
                search_results = self.dataset.search(
                    embedding=query_embedding,
                    k=top_k,
                    filter={"document_id": document_id}
                )
            else:
                search_results = self.dataset.search(
                    embedding=query_embedding,
                    k=top_k
                )

            # Format results
            results = []
            for result in search_results:
                results.append({
                    'text': result['text'],
                    'similarity_score': result['score'],
                    'metadata': result['metadata'],
                    'page_number': result['page_number'],
                    'document_id': result['document_id']
                })

            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def get_document_chunks(self, document_id: str) -> List[PDFChunk]:
        """
        Retrieve all chunks for a specific document.

        Args:
            document_id (str): Document identifier

        Returns:
            List[PDFChunk]: Document chunks
        """
        if not self.dataset:
            logger.warning("DeepLake dataset not available")
            return []

        try:
            # Query dataset for document chunks
            results = self.dataset.filter(lambda x: x['document_id'] == document_id)

            chunks = []
            for result in results:
                chunk = PDFChunk(
                    content=result['text'],
                    page_number=result['page_number'],
                    chunk_index=result['chunk_index'],
                    start_char=0,
                    end_char=len(result['text']),
                    chunk_type=result['metadata'].get('chunk_type', 'text'),
                    bbox=None,
                    font_info=None,
                    embedding=result['embedding'],
                    metadata=result['metadata']
                )
                chunks.append(chunk)

            return sorted(chunks, key=lambda x: x.chunk_index)

        except Exception as e:
            logger.error(f"Document chunk retrieval failed: {e}")
            return []
