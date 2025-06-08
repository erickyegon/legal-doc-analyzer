"""
Multimodal Extraction Agent for the Legal Intelligence Platform.

This agent handles complex document analysis including diagrams, tables, 
signatures, and layout-aware parsing using advanced computer vision and OCR.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import logging
import os
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from dataclasses import dataclass

# LangChain imports
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from langchain.schema import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Computer vision and OCR
import easyocr
import pytesseract
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import dict_to_elements
import layoutparser as lp
import tabula
import camelot

# Custom imports
from app.euri_client import euri_chat_completion
from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Data class for extraction results."""
    text_content: str
    tables: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    signatures: List[Dict[str, Any]]
    diagrams: List[Dict[str, Any]]
    layout_elements: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class AgentState(TypedDict):
    """State for the multimodal extraction agent."""
    messages: List[BaseMessage]
    document_path: str
    extraction_results: Optional[ExtractionResult]
    current_step: str
    errors: List[str]


class MultimodalExtractionAgent:
    """
    Advanced multimodal extraction agent for legal documents.
    
    This agent uses computer vision, OCR, and layout analysis to extract
    comprehensive information from legal documents including:
    - Text content with layout preservation
    - Tables and tabular data (payment schedules, terms, etc.)
    - Images and diagrams
    - Signatures and stamps
    - Document structure and layout elements
    """
    
    def __init__(self):
        """Initialize the multimodal extraction agent."""
        self.ocr_reader = easyocr.Reader(['en'])
        self.layout_model = self._initialize_layout_model()
        self.graph = self._build_extraction_graph()
        
    def _initialize_layout_model(self):
        """Initialize the layout analysis model."""
        try:
            # Initialize LayoutParser model for document layout analysis
            model = lp.Detectron2LayoutModel(
                'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            )
            return model
        except Exception as e:
            logger.warning(f"Failed to initialize layout model: {e}")
            return None
    
    def _build_extraction_graph(self) -> StateGraph:
        """Build the LangGraph workflow for multimodal extraction."""
        
        def extract_text_content(state: AgentState) -> AgentState:
            """Extract text content from document."""
            try:
                document_path = state["document_path"]
                text_content = self._extract_text_with_layout(document_path)
                
                if not state.get("extraction_results"):
                    state["extraction_results"] = ExtractionResult(
                        text_content="", tables=[], images=[], 
                        signatures=[], diagrams=[], layout_elements=[], metadata={}
                    )
                
                state["extraction_results"].text_content = text_content
                state["current_step"] = "text_extracted"
                
            except Exception as e:
                state["errors"].append(f"Text extraction failed: {str(e)}")
                logger.error(f"Text extraction error: {e}")
            
            return state
        
        def extract_tables(state: AgentState) -> AgentState:
            """Extract tables from document."""
            try:
                document_path = state["document_path"]
                tables = self._extract_tables(document_path)
                
                if state.get("extraction_results"):
                    state["extraction_results"].tables = tables
                
                state["current_step"] = "tables_extracted"
                
            except Exception as e:
                state["errors"].append(f"Table extraction failed: {str(e)}")
                logger.error(f"Table extraction error: {e}")
            
            return state
        
        def extract_images_and_diagrams(state: AgentState) -> AgentState:
            """Extract images and diagrams from document."""
            try:
                document_path = state["document_path"]
                images, diagrams = self._extract_images_and_diagrams(document_path)
                
                if state.get("extraction_results"):
                    state["extraction_results"].images = images
                    state["extraction_results"].diagrams = diagrams
                
                state["current_step"] = "images_extracted"
                
            except Exception as e:
                state["errors"].append(f"Image extraction failed: {str(e)}")
                logger.error(f"Image extraction error: {e}")
            
            return state
        
        def detect_signatures(state: AgentState) -> AgentState:
            """Detect and extract signatures from document."""
            try:
                document_path = state["document_path"]
                signatures = self._detect_signatures(document_path)
                
                if state.get("extraction_results"):
                    state["extraction_results"].signatures = signatures
                
                state["current_step"] = "signatures_detected"
                
            except Exception as e:
                state["errors"].append(f"Signature detection failed: {str(e)}")
                logger.error(f"Signature detection error: {e}")
            
            return state
        
        def analyze_layout(state: AgentState) -> AgentState:
            """Analyze document layout and structure."""
            try:
                document_path = state["document_path"]
                layout_elements = self._analyze_layout(document_path)
                
                if state.get("extraction_results"):
                    state["extraction_results"].layout_elements = layout_elements
                
                state["current_step"] = "layout_analyzed"
                
            except Exception as e:
                state["errors"].append(f"Layout analysis failed: {str(e)}")
                logger.error(f"Layout analysis error: {e}")
            
            return state
        
        def finalize_extraction(state: AgentState) -> AgentState:
            """Finalize extraction and generate metadata."""
            try:
                if state.get("extraction_results"):
                    metadata = self._generate_metadata(state["extraction_results"])
                    state["extraction_results"].metadata = metadata
                
                state["current_step"] = "completed"
                
            except Exception as e:
                state["errors"].append(f"Finalization failed: {str(e)}")
                logger.error(f"Finalization error: {e}")
            
            return state
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("extract_text", extract_text_content)
        workflow.add_node("extract_tables", extract_tables)
        workflow.add_node("extract_images", extract_images_and_diagrams)
        workflow.add_node("detect_signatures", detect_signatures)
        workflow.add_node("analyze_layout", analyze_layout)
        workflow.add_node("finalize", finalize_extraction)
        
        # Add edges
        workflow.set_entry_point("extract_text")
        workflow.add_edge("extract_text", "extract_tables")
        workflow.add_edge("extract_tables", "extract_images")
        workflow.add_edge("extract_images", "detect_signatures")
        workflow.add_edge("detect_signatures", "analyze_layout")
        workflow.add_edge("analyze_layout", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _extract_text_with_layout(self, document_path: str) -> str:
        """
        Extract text content while preserving layout information.
        
        Args:
            document_path (str): Path to the document
            
        Returns:
            str: Extracted text with layout preservation
        """
        try:
            # Use pdfplumber for layout-aware text extraction
            with pdfplumber.open(document_path) as pdf:
                text_content = []
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text with position information
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                
                return "\n\n".join(text_content)
                
        except Exception as e:
            logger.error(f"Layout-aware text extraction failed: {e}")
            # Fallback to basic text extraction
            return self._basic_text_extraction(document_path)
    
    def _basic_text_extraction(self, document_path: str) -> str:
        """Fallback basic text extraction."""
        try:
            doc = fitz.open(document_path)
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")
            
            doc.close()
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Basic text extraction failed: {e}")
            return ""
    
    def _extract_tables(self, document_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from document using multiple methods.
        
        Args:
            document_path (str): Path to the document
            
        Returns:
            List[Dict]: Extracted tables with metadata
        """
        tables = []
        
        try:
            # Method 1: Use Camelot for high-quality table extraction
            camelot_tables = camelot.read_pdf(document_path, pages='all')
            
            for i, table in enumerate(camelot_tables):
                table_data = {
                    'id': f'camelot_table_{i}',
                    'method': 'camelot',
                    'page': table.page,
                    'data': table.df.to_dict('records'),
                    'shape': table.shape,
                    'accuracy': table.accuracy if hasattr(table, 'accuracy') else None,
                    'raw_data': table.df.to_csv(index=False)
                }
                tables.append(table_data)
                
        except Exception as e:
            logger.warning(f"Camelot table extraction failed: {e}")
        
        try:
            # Method 2: Use Tabula as fallback
            tabula_tables = tabula.read_pdf(document_path, pages='all', multiple_tables=True)
            
            for i, df in enumerate(tabula_tables):
                if not df.empty:
                    table_data = {
                        'id': f'tabula_table_{i}',
                        'method': 'tabula',
                        'page': i + 1,  # Approximate page number
                        'data': df.to_dict('records'),
                        'shape': df.shape,
                        'raw_data': df.to_csv(index=False)
                    }
                    tables.append(table_data)
                    
        except Exception as e:
            logger.warning(f"Tabula table extraction failed: {e}")
        
        try:
            # Method 3: Use pdfplumber for table detection
            with pdfplumber.open(document_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for i, table in enumerate(page_tables):
                        if table:
                            # Convert to DataFrame for consistency
                            df = pd.DataFrame(table[1:], columns=table[0])
                            
                            table_data = {
                                'id': f'pdfplumber_table_{page_num}_{i}',
                                'method': 'pdfplumber',
                                'page': page_num + 1,
                                'data': df.to_dict('records'),
                                'shape': df.shape,
                                'raw_data': df.to_csv(index=False)
                            }
                            tables.append(table_data)
                            
        except Exception as e:
            logger.warning(f"PDFPlumber table extraction failed: {e}")
        
        # Remove duplicates and rank by quality
        tables = self._deduplicate_and_rank_tables(tables)
        
        return tables
    
    def _deduplicate_and_rank_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tables and rank by quality."""
        if not tables:
            return tables
        
        # Simple deduplication based on shape and content similarity
        unique_tables = []
        
        for table in tables:
            is_duplicate = False
            
            for existing_table in unique_tables:
                # Check if tables are similar (same shape and similar content)
                if (table['shape'] == existing_table['shape'] and 
                    self._tables_are_similar(table['data'], existing_table['data'])):
                    
                    # Keep the one with higher quality (prefer camelot > pdfplumber > tabula)
                    quality_order = {'camelot': 3, 'pdfplumber': 2, 'tabula': 1}
                    
                    if quality_order.get(table['method'], 0) > quality_order.get(existing_table['method'], 0):
                        # Replace with higher quality table
                        unique_tables.remove(existing_table)
                        unique_tables.append(table)
                    
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tables.append(table)
        
        return unique_tables
    
    def _tables_are_similar(self, table1_data: List[Dict], table2_data: List[Dict], threshold: float = 0.8) -> bool:
        """Check if two tables are similar based on content."""
        if len(table1_data) != len(table2_data):
            return False
        
        if not table1_data or not table2_data:
            return False
        
        # Compare first few rows for similarity
        rows_to_compare = min(3, len(table1_data))
        similar_rows = 0
        
        for i in range(rows_to_compare):
            row1_str = str(table1_data[i]).lower()
            row2_str = str(table2_data[i]).lower()
            
            # Simple string similarity check
            if row1_str == row2_str:
                similar_rows += 1
        
        similarity = similar_rows / rows_to_compare
        return similarity >= threshold

    def _extract_images_and_diagrams(self, document_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract images and diagrams from document.

        Args:
            document_path (str): Path to the document

        Returns:
            Tuple[List[Dict], List[Dict]]: Images and diagrams with metadata
        """
        images = []
        diagrams = []

        try:
            doc = fitz.open(document_path)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)

                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")

                            # Save image temporarily for analysis
                            temp_img_path = f"/tmp/extracted_img_{page_num}_{img_index}.png"
                            with open(temp_img_path, "wb") as f:
                                f.write(img_data)

                            # Analyze image to determine if it's a diagram or regular image
                            is_diagram = self._is_diagram(temp_img_path)

                            img_metadata = {
                                'id': f'img_{page_num}_{img_index}',
                                'page': page_num + 1,
                                'xref': xref,
                                'width': pix.width,
                                'height': pix.height,
                                'colorspace': pix.colorspace.name if pix.colorspace else 'unknown',
                                'temp_path': temp_img_path,
                                'size_bytes': len(img_data)
                            }

                            if is_diagram:
                                # Extract text from diagram using OCR
                                diagram_text = self._extract_text_from_image(temp_img_path)
                                img_metadata['extracted_text'] = diagram_text
                                img_metadata['type'] = 'diagram'
                                diagrams.append(img_metadata)
                            else:
                                img_metadata['type'] = 'image'
                                images.append(img_metadata)

                        pix = None

                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                        continue

            doc.close()

        except Exception as e:
            logger.error(f"Image extraction failed: {e}")

        return images, diagrams

    def _is_diagram(self, image_path: str) -> bool:
        """
        Determine if an image is likely a diagram/chart.

        Args:
            image_path (str): Path to the image

        Returns:
            bool: True if image is likely a diagram
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return False

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect lines (diagrams often have many lines)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

            # Detect text regions
            text_regions = self._detect_text_regions(gray)

            # Heuristics for diagram detection
            line_count = len(lines) if lines is not None else 0
            text_region_count = len(text_regions)

            # If many lines and some text, likely a diagram
            if line_count > 10 and text_region_count > 0:
                return True

            # Check for geometric shapes
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            geometric_shapes = 0

            for contour in contours:
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Count shapes with 3-8 vertices (triangles, rectangles, etc.)
                if 3 <= len(approx) <= 8:
                    geometric_shapes += 1

            # If many geometric shapes, likely a diagram
            if geometric_shapes > 5:
                return True

            return False

        except Exception as e:
            logger.warning(f"Diagram detection failed for {image_path}: {e}")
            return False

    def _detect_text_regions(self, gray_image) -> List[Dict[str, Any]]:
        """Detect text regions in an image."""
        try:
            # Use MSER (Maximally Stable Extremal Regions) for text detection
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray_image)

            text_regions = []
            for region in regions:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))

                # Filter by size (text regions should be reasonable size)
                if 10 < w < 200 and 5 < h < 50:
                    text_regions.append({
                        'x': x, 'y': y, 'width': w, 'height': h,
                        'area': w * h
                    })

            return text_regions

        except Exception as e:
            logger.warning(f"Text region detection failed: {e}")
            return []

    def _extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using OCR.

        Args:
            image_path (str): Path to the image

        Returns:
            str: Extracted text
        """
        try:
            # Try EasyOCR first
            results = self.ocr_reader.readtext(image_path)
            text_parts = [result[1] for result in results if result[2] > 0.5]  # Confidence > 0.5

            if text_parts:
                return " ".join(text_parts)

            # Fallback to Tesseract
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            return text.strip()

        except Exception as e:
            logger.warning(f"OCR failed for {image_path}: {e}")
            return ""

    def _detect_signatures(self, document_path: str) -> List[Dict[str, Any]]:
        """
        Detect signatures in the document.

        Args:
            document_path (str): Path to the document

        Returns:
            List[Dict]: Detected signatures with metadata
        """
        signatures = []

        try:
            doc = fitz.open(document_path)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                # Convert page to image for signature detection
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
                img_data = pix.tobytes("png")

                # Save temporary image
                temp_page_path = f"/tmp/page_{page_num}.png"
                with open(temp_page_path, "wb") as f:
                    f.write(img_data)

                # Detect signatures using computer vision
                page_signatures = self._detect_signatures_in_image(temp_page_path, page_num + 1)
                signatures.extend(page_signatures)

                # Clean up
                if os.path.exists(temp_page_path):
                    os.remove(temp_page_path)

            doc.close()

        except Exception as e:
            logger.error(f"Signature detection failed: {e}")

        return signatures

    def _detect_signatures_in_image(self, image_path: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Detect signatures in a page image.

        Args:
            image_path (str): Path to the page image
            page_num (int): Page number

        Returns:
            List[Dict]: Detected signatures
        """
        signatures = []

        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return signatures

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Signature detection heuristics
            # 1. Look for handwritten-like regions (irregular contours)
            # 2. Look for regions with specific aspect ratios
            # 3. Look for regions near signature keywords

            # Find contours
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Look for signature keywords
            signature_keywords = ['signature', 'sign', 'signed', 'date', '/s/', 'electronically signed']
            page_text = self._extract_text_from_image(image_path).lower()

            has_signature_keywords = any(keyword in page_text for keyword in signature_keywords)

            for i, contour in enumerate(contours):
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Signature heuristics
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)

                # Typical signature characteristics:
                # - Aspect ratio between 2:1 and 6:1
                # - Reasonable size (not too small or too large)
                # - Irregular shape (low solidity)

                if (2 <= aspect_ratio <= 6 and
                    1000 <= area <= 50000 and
                    w > 50 and h > 10):

                    # Calculate solidity (area / convex hull area)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0

                    # Signatures tend to have lower solidity (more irregular)
                    if solidity < 0.8:
                        # Extract the signature region
                        signature_region = gray[y:y+h, x:x+w]

                        # Additional validation using OCR
                        signature_text = self._extract_text_from_signature_region(signature_region)

                        signature_data = {
                            'id': f'signature_{page_num}_{i}',
                            'page': page_num,
                            'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'solidity': solidity,
                            'confidence': self._calculate_signature_confidence(
                                aspect_ratio, area, solidity, has_signature_keywords, signature_text
                            ),
                            'extracted_text': signature_text,
                            'near_keywords': has_signature_keywords
                        }

                        # Only include if confidence is reasonable
                        if signature_data['confidence'] > 0.3:
                            signatures.append(signature_data)

        except Exception as e:
            logger.warning(f"Signature detection failed for {image_path}: {e}")

        return signatures

    def _extract_text_from_signature_region(self, signature_region) -> str:
        """Extract text from a signature region."""
        try:
            # Convert to PIL Image
            pil_img = Image.fromarray(signature_region)

            # Use Tesseract with specific config for handwriting
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
            text = pytesseract.image_to_string(pil_img, config=custom_config)

            return text.strip()

        except Exception as e:
            logger.warning(f"Signature text extraction failed: {e}")
            return ""

    def _calculate_signature_confidence(self, aspect_ratio: float, area: float,
                                      solidity: float, has_keywords: bool,
                                      signature_text: str) -> float:
        """Calculate confidence score for signature detection."""
        confidence = 0.0

        # Aspect ratio score
        if 2 <= aspect_ratio <= 6:
            confidence += 0.3

        # Area score
        if 1000 <= area <= 50000:
            confidence += 0.2

        # Solidity score (lower is better for signatures)
        if solidity < 0.6:
            confidence += 0.3
        elif solidity < 0.8:
            confidence += 0.1

        # Keyword proximity
        if has_keywords:
            confidence += 0.2

        # Text analysis
        if signature_text:
            # Check if text looks like a name (2-4 words, proper case)
            words = signature_text.split()
            if 1 <= len(words) <= 4 and any(word.istitle() for word in words):
                confidence += 0.2

        return min(confidence, 1.0)

    def _analyze_layout(self, document_path: str) -> List[Dict[str, Any]]:
        """
        Analyze document layout and structure.

        Args:
            document_path (str): Path to the document

        Returns:
            List[Dict]: Layout elements with metadata
        """
        layout_elements = []

        try:
            if self.layout_model is None:
                logger.warning("Layout model not available, using fallback method")
                return self._fallback_layout_analysis(document_path)

            doc = fitz.open(document_path)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")

                # Save temporary image
                temp_page_path = f"/tmp/layout_page_{page_num}.png"
                with open(temp_page_path, "wb") as f:
                    f.write(img_data)

                # Analyze layout using LayoutParser
                image = cv2.imread(temp_page_path)
                layout = self.layout_model.detect(image)

                for i, element in enumerate(layout):
                    element_data = {
                        'id': f'layout_{page_num}_{i}',
                        'page': page_num + 1,
                        'type': element.type,
                        'bbox': {
                            'x': element.block.x_1,
                            'y': element.block.y_1,
                            'width': element.block.width,
                            'height': element.block.height
                        },
                        'confidence': element.score,
                        'area': element.block.area
                    }

                    # Extract text from the element region if it's a text element
                    if element.type in ['Text', 'Title', 'List']:
                        x1, y1, x2, y2 = int(element.block.x_1), int(element.block.y_1), int(element.block.x_2), int(element.block.y_2)
                        element_region = image[y1:y2, x1:x2]
                        element_text = self._extract_text_from_image_region(element_region)
                        element_data['text'] = element_text

                    layout_elements.append(element_data)

                # Clean up
                if os.path.exists(temp_page_path):
                    os.remove(temp_page_path)

            doc.close()

        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            return self._fallback_layout_analysis(document_path)

        return layout_elements

    def _fallback_layout_analysis(self, document_path: str) -> List[Dict[str, Any]]:
        """Fallback layout analysis using basic methods."""
        layout_elements = []

        try:
            with pdfplumber.open(document_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Get text objects with positions
                    chars = page.chars

                    if chars:
                        # Group characters into text blocks
                        text_blocks = self._group_chars_into_blocks(chars)

                        for i, block in enumerate(text_blocks):
                            element_data = {
                                'id': f'fallback_layout_{page_num}_{i}',
                                'page': page_num + 1,
                                'type': 'Text',
                                'bbox': block['bbox'],
                                'text': block['text'],
                                'confidence': 0.8  # Default confidence for fallback
                            }
                            layout_elements.append(element_data)

        except Exception as e:
            logger.error(f"Fallback layout analysis failed: {e}")

        return layout_elements

    def _group_chars_into_blocks(self, chars: List[Dict]) -> List[Dict[str, Any]]:
        """Group characters into text blocks based on proximity."""
        if not chars:
            return []

        # Sort characters by position
        sorted_chars = sorted(chars, key=lambda c: (c['top'], c['x0']))

        blocks = []
        current_block = {
            'chars': [sorted_chars[0]],
            'text': sorted_chars[0]['text'],
            'bbox': {
                'x': sorted_chars[0]['x0'],
                'y': sorted_chars[0]['top'],
                'width': sorted_chars[0]['width'],
                'height': sorted_chars[0]['height']
            }
        }

        for char in sorted_chars[1:]:
            # Check if character belongs to current block
            # (similar y-position and reasonable x-distance)
            last_char = current_block['chars'][-1]

            y_diff = abs(char['top'] - last_char['top'])
            x_diff = char['x0'] - last_char['x1']

            if y_diff < 5 and x_diff < 20:  # Same line, reasonable gap
                # Add to current block
                current_block['chars'].append(char)
                current_block['text'] += char['text']

                # Update bounding box
                current_block['bbox']['width'] = char['x1'] - current_block['bbox']['x']
                current_block['bbox']['height'] = max(
                    current_block['bbox']['height'],
                    char['bottom'] - current_block['bbox']['y']
                )
            else:
                # Start new block
                if len(current_block['text'].strip()) > 0:
                    blocks.append(current_block)

                current_block = {
                    'chars': [char],
                    'text': char['text'],
                    'bbox': {
                        'x': char['x0'],
                        'y': char['top'],
                        'width': char['width'],
                        'height': char['height']
                    }
                }

        # Add last block
        if len(current_block['text'].strip()) > 0:
            blocks.append(current_block)

        return blocks

    def _extract_text_from_image_region(self, image_region) -> str:
        """Extract text from a specific image region."""
        try:
            # Save region as temporary image
            temp_path = "/tmp/temp_region.png"
            cv2.imwrite(temp_path, image_region)

            # Extract text
            text = self._extract_text_from_image(temp_path)

            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return text

        except Exception as e:
            logger.warning(f"Text extraction from image region failed: {e}")
            return ""

    def _generate_metadata(self, extraction_results: ExtractionResult) -> Dict[str, Any]:
        """Generate comprehensive metadata for extraction results."""
        metadata = {
            'extraction_summary': {
                'text_length': len(extraction_results.text_content),
                'table_count': len(extraction_results.tables),
                'image_count': len(extraction_results.images),
                'diagram_count': len(extraction_results.diagrams),
                'signature_count': len(extraction_results.signatures),
                'layout_element_count': len(extraction_results.layout_elements)
            },
            'quality_metrics': {
                'has_tables': len(extraction_results.tables) > 0,
                'has_images': len(extraction_results.images) > 0,
                'has_signatures': len(extraction_results.signatures) > 0,
                'layout_analyzed': len(extraction_results.layout_elements) > 0
            },
            'extraction_methods': {
                'text_extraction': 'pdfplumber_with_layout',
                'table_extraction': ['camelot', 'tabula', 'pdfplumber'],
                'image_extraction': 'pymupdf',
                'signature_detection': 'opencv_heuristics',
                'layout_analysis': 'layoutparser' if self.layout_model else 'fallback'
            }
        }

        # Add table-specific metadata
        if extraction_results.tables:
            table_methods = list(set(table['method'] for table in extraction_results.tables))
            metadata['table_extraction_methods'] = table_methods

            # Calculate average table accuracy
            accuracies = [table.get('accuracy', 0) for table in extraction_results.tables if table.get('accuracy')]
            if accuracies:
                metadata['average_table_accuracy'] = sum(accuracies) / len(accuracies)

        # Add signature confidence scores
        if extraction_results.signatures:
            confidences = [sig['confidence'] for sig in extraction_results.signatures]
            metadata['signature_confidence'] = {
                'average': sum(confidences) / len(confidences),
                'max': max(confidences),
                'min': min(confidences)
            }

        return metadata

    async def extract(self, document_path: str) -> ExtractionResult:
        """
        Main extraction method using the LangGraph workflow.

        Args:
            document_path (str): Path to the document to extract from

        Returns:
            ExtractionResult: Comprehensive extraction results
        """
        try:
            # Initialize state
            initial_state = AgentState(
                messages=[],
                document_path=document_path,
                extraction_results=None,
                current_step="initialized",
                errors=[]
            )

            # Run the extraction workflow
            final_state = await self.graph.ainvoke(initial_state)

            # Return results
            if final_state.get("extraction_results"):
                return final_state["extraction_results"]
            else:
                # Return empty results if extraction failed
                return ExtractionResult(
                    text_content="",
                    tables=[],
                    images=[],
                    signatures=[],
                    diagrams=[],
                    layout_elements=[],
                    metadata={"errors": final_state.get("errors", [])}
                )

        except Exception as e:
            logger.error(f"Multimodal extraction failed: {e}")
            return ExtractionResult(
                text_content="",
                tables=[],
                images=[],
                signatures=[],
                diagrams=[],
                layout_elements=[],
                metadata={"errors": [str(e)]}
            )
