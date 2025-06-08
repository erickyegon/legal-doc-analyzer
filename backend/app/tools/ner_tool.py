"""
Named Entity Recognition (NER) Tool for the Legal Intelligence Platform.

This tool provides advanced NER capabilities using custom-trained spaCy models
and OpenAI function calls for legal entity extraction and classification.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

# NLP libraries
import spacy
from spacy import displacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# LangChain imports
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Custom imports
from app.euri_client import euri_chat_completion
from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Legal entity types for classification."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    LAW = "LAW"
    CASE = "CASE"
    COURT = "COURT"
    JUDGE = "JUDGE"
    LAWYER = "LAWYER"
    CONTRACT_TERM = "CONTRACT_TERM"
    LEGAL_CONCEPT = "LEGAL_CONCEPT"
    JURISDICTION = "JURISDICTION"
    STATUTE = "STATUTE"
    REGULATION = "REGULATION"
    CLAUSE_TYPE = "CLAUSE_TYPE"
    PARTY_ROLE = "PARTY_ROLE"
    DOCUMENT_TYPE = "DOCUMENT_TYPE"
    LEGAL_ACTION = "LEGAL_ACTION"


@dataclass
class Entity:
    """Data class for extracted entities."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    context: str
    metadata: Dict[str, Any]
    extraction_method: str


@dataclass
class NERResult:
    """Result of NER extraction."""
    entities: List[Entity]
    entity_counts: Dict[str, int]
    relationships: List[Dict[str, Any]]
    confidence_score: float
    processing_stats: Dict[str, Any]


class EntityExtraction(BaseModel):
    """Pydantic model for OpenAI function calling."""
    entities: List[Dict[str, Any]] = Field(description="List of extracted entities")
    relationships: List[Dict[str, Any]] = Field(description="Relationships between entities")
    confidence: float = Field(description="Overall confidence score")


class NERTool(BaseTool):
    """
    Advanced Named Entity Recognition Tool for legal documents.
    
    This tool provides:
    - Custom-trained spaCy models for legal entities
    - OpenAI function calls for complex entity extraction
    - Legal-specific entity types and relationships
    - Confidence scoring and validation
    - Entity linking and normalization
    - Relationship extraction between entities
    """
    
    name = "ner_tool"
    description = "Extract and classify named entities from legal documents"
    
    def __init__(self, 
                 spacy_model: str = "en_core_web_sm",
                 custom_model_path: Optional[str] = None):
        """
        Initialize the NER Tool.
        
        Args:
            spacy_model (str): Base spaCy model to use
            custom_model_path (str, optional): Path to custom trained model
        """
        super().__init__()
        
        # Initialize spaCy model
        self.nlp = None
        self._initialize_spacy_model(spacy_model, custom_model_path)
        
        # Initialize NLTK components
        self._initialize_nltk()
        
        # Legal entity patterns
        self.legal_patterns = self._initialize_legal_patterns()
        
        # Entity validation rules
        self.validation_rules = self._initialize_validation_rules()
        
        # Output parser for OpenAI function calls
        self.output_parser = JsonOutputParser()
    
    def _initialize_spacy_model(self, model_name: str, custom_path: Optional[str]):
        """Initialize spaCy model with custom legal entities."""
        try:
            if custom_path:
                # Load custom trained model
                self.nlp = spacy.load(custom_path)
                logger.info(f"Loaded custom spaCy model from {custom_path}")
            else:
                # Load base model and add custom components
                self.nlp = spacy.load(model_name)
                self._add_legal_entity_ruler()
                logger.info(f"Loaded base spaCy model: {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            self.nlp = None
    
    def _add_legal_entity_ruler(self):
        """Add custom entity ruler for legal entities."""
        if not self.nlp:
            return
        
        try:
            # Add entity ruler to pipeline
            if "entity_ruler" not in self.nlp.pipe_names:
                ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            else:
                ruler = self.nlp.get_pipe("entity_ruler")
            
            # Define legal entity patterns
            legal_patterns = [
                # Court patterns
                {"label": "COURT", "pattern": [{"LOWER": {"IN": ["supreme", "district", "circuit", "appellate"}}], {"LOWER": "court"}]},
                {"label": "COURT", "pattern": [{"TEXT": {"REGEX": r".*\s+v\.?\s+.*"}}]},  # Case names
                
                # Legal concepts
                {"label": "LEGAL_CONCEPT", "pattern": [{"LOWER": {"IN": ["breach", "contract", "negligence", "liability", "damages"]}}]},
                {"label": "LEGAL_CONCEPT", "pattern": [{"LOWER": "force"}, {"LOWER": "majeure"}]},
                
                # Contract terms
                {"label": "CONTRACT_TERM", "pattern": [{"LOWER": {"IN": ["termination", "renewal", "assignment", "indemnification"]}}]},
                
                # Legal actions
                {"label": "LEGAL_ACTION", "pattern": [{"LOWER": {"IN": ["sue", "sued", "lawsuit", "litigation", "arbitration", "mediation"]}}]},
                
                # Jurisdictions
                {"label": "JURISDICTION", "pattern": [{"LOWER": {"IN": ["federal", "state", "local", "municipal"]}}]},
                
                # Document types
                {"label": "DOCUMENT_TYPE", "pattern": [{"LOWER": {"IN": ["contract", "agreement", "policy", "statute", "regulation", "ordinance"]}}]},
            ]
            
            ruler.add_patterns(legal_patterns)
            logger.info("Added legal entity patterns to spaCy model")
            
        except Exception as e:
            logger.error(f"Failed to add entity ruler: {e}")
    
    def _initialize_nltk(self):
        """Initialize NLTK components."""
        try:
            # Download required NLTK data
            nltk_downloads = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
            for download in nltk_downloads:
                try:
                    nltk.data.find(f'tokenizers/{download}')
                except LookupError:
                    nltk.download(download, quiet=True)
            
            logger.info("Initialized NLTK components")
            
        except Exception as e:
            logger.warning(f"NLTK initialization failed: {e}")
    
    def _initialize_legal_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for legal entity extraction."""
        return {
            "case_citations": [
                r'\d+\s+[A-Z][a-z]*\.?\s*\d+d?\s*\d+',  # 123 F.3d 456
                r'\d+\s+U\.S\.?\s+\d+',  # 123 U.S. 456
                r'\d+\s+S\.?\s*Ct\.?\s+\d+',  # 123 S. Ct. 456
            ],
            "statutes": [
                r'\d+\s+U\.S\.C\.?\s*§?\s*\d+',  # 42 U.S.C. § 1983
                r'§\s*\d+[\.\d]*',  # § 123.45
                r'Section\s+\d+[\.\d]*',  # Section 123.45
            ],
            "dates": [
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            ],
            "money": [
                r'\$[\d,]+(?:\.\d{2})?',
                r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|EUR|GBP)\b',
            ],
            "percentages": [
                r'\b\d+(?:\.\d+)?%\b',
                r'\b\d+(?:\.\d+)?\s*percent\b',
            ],
            "party_roles": [
                r'\b(?:plaintiff|defendant|appellant|appellee|petitioner|respondent|claimant|grantor|grantee|lessor|lessee|licensor|licensee|buyer|seller|contractor|subcontractor)\b',
            ],
            "legal_documents": [
                r'\b(?:contract|agreement|lease|license|deed|will|trust|policy|statute|regulation|ordinance|rule|order|judgment|decree|injunction)\b',
            ]
        }
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize entity validation rules."""
        return {
            "min_confidence": 0.5,
            "min_entity_length": 2,
            "max_entity_length": 100,
            "blacklist_words": ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"],
            "required_pos_tags": {
                "PERSON": ["NNP", "NNPS"],
                "ORGANIZATION": ["NNP", "NNPS"],
                "LOCATION": ["NNP", "NNPS"]
            }
        }
    
    def _run(self, text: str, **kwargs) -> str:
        """
        Run the NER tool.
        
        Args:
            text (str): Text to extract entities from
            **kwargs: Additional arguments
            
        Returns:
            str: Summary of extracted entities
        """
        try:
            result = self.extract_entities(text, **kwargs)
            
            # Format summary
            summary = f"Extracted {len(result.entities)} entities:\n"
            
            for entity_type, count in result.entity_counts.items():
                summary += f"- {entity_type}: {count}\n"
            
            summary += f"\nConfidence Score: {result.confidence_score:.2f}"
            
            if result.relationships:
                summary += f"\nRelationships: {len(result.relationships)}"
            
            return summary
            
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return f"NER extraction failed: {str(e)}"
    
    def extract_entities(self,
                        text: str,
                        use_spacy: bool = True,
                        use_openai: bool = True,
                        use_regex: bool = True,
                        extract_relationships: bool = True) -> NERResult:
        """
        Extract entities using multiple methods.
        
        Args:
            text (str): Text to process
            use_spacy (bool): Use spaCy NER
            use_openai (bool): Use OpenAI function calls
            use_regex (bool): Use regex patterns
            extract_relationships (bool): Extract entity relationships
            
        Returns:
            NERResult: Comprehensive NER results
        """
        start_time = datetime.now()
        all_entities = []
        
        try:
            # Method 1: spaCy NER
            if use_spacy and self.nlp:
                spacy_entities = self._extract_with_spacy(text)
                all_entities.extend(spacy_entities)
            
            # Method 2: OpenAI function calls
            if use_openai:
                openai_entities = self._extract_with_openai(text)
                all_entities.extend(openai_entities)
            
            # Method 3: Regex patterns
            if use_regex:
                regex_entities = self._extract_with_regex(text)
                all_entities.extend(regex_entities)
            
            # Deduplicate and merge entities
            merged_entities = self._merge_entities(all_entities)
            
            # Validate entities
            validated_entities = self._validate_entities(merged_entities, text)
            
            # Extract relationships
            relationships = []
            if extract_relationships:
                relationships = self._extract_relationships(validated_entities, text)
            
            # Calculate statistics
            entity_counts = self._calculate_entity_counts(validated_entities)
            confidence_score = self._calculate_overall_confidence(validated_entities)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            processing_stats = {
                "processing_time": processing_time,
                "methods_used": {
                    "spacy": use_spacy and self.nlp is not None,
                    "openai": use_openai,
                    "regex": use_regex
                },
                "total_entities_found": len(all_entities),
                "entities_after_validation": len(validated_entities)
            }
            
            return NERResult(
                entities=validated_entities,
                entity_counts=entity_counts,
                relationships=relationships,
                confidence_score=confidence_score,
                processing_stats=processing_stats
            )
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return NERResult(
                entities=[],
                entity_counts={},
                relationships=[],
                confidence_score=0.0,
                processing_stats={"error": str(e)}
            )
    
    def _extract_with_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy."""
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Get context (surrounding words)
                start_context = max(0, ent.start - 5)
                end_context = min(len(doc), ent.end + 5)
                context = doc[start_context:end_context].text
                
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8,  # Default confidence for spaCy
                    context=context,
                    metadata={
                        "spacy_kb_id": ent.kb_id_ if hasattr(ent, 'kb_id_') else None,
                        "spacy_sentiment": getattr(ent, 'sentiment', None)
                    },
                    extraction_method="spacy"
                )
                entities.append(entity)
            
            logger.info(f"spaCy extracted {len(entities)} entities")
            
        except Exception as e:
            logger.error(f"spaCy extraction failed: {e}")
        
        return entities
    
    def _extract_with_openai(self, text: str) -> List[Entity]:
        """Extract entities using OpenAI function calls."""
        entities = []
        
        try:
            # Prepare prompt for entity extraction
            prompt = PromptTemplate(
                input_variables=["text"],
                template="""
                Extract all named entities from the following legal text. Focus on:
                - People (lawyers, judges, parties)
                - Organizations (law firms, companies, courts)
                - Legal concepts (laws, cases, statutes)
                - Dates and monetary amounts
                - Locations and jurisdictions
                
                Text: {text}
                
                Return the entities in JSON format with the following structure:
                {{
                    "entities": [
                        {{
                            "text": "entity text",
                            "label": "entity type",
                            "start": start_position,
                            "end": end_position,
                            "confidence": confidence_score
                        }}
                    ],
                    "relationships": [
                        {{
                            "entity1": "first entity",
                            "entity2": "second entity", 
                            "relationship": "relationship type"
                        }}
                    ],
                    "confidence": overall_confidence
                }}
                """
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a legal NER expert. Extract entities accurately and provide confidence scores."
                },
                {
                    "role": "user",
                    "content": prompt.format(text=text[:4000])  # Limit text length
                }
            ]
            
            response = euri_chat_completion(
                messages=messages,
                model=settings.euri_model,
                temperature=0.1,
                max_tokens=2000
            )
            
            # Parse response
            try:
                parsed_response = json.loads(response)
                
                for ent_data in parsed_response.get("entities", []):
                    # Get context
                    start = ent_data.get("start", 0)
                    end = ent_data.get("end", len(ent_data.get("text", "")))
                    context_start = max(0, start - 50)
                    context_end = min(len(text), end + 50)
                    context = text[context_start:context_end]
                    
                    entity = Entity(
                        text=ent_data.get("text", ""),
                        label=ent_data.get("label", "UNKNOWN"),
                        start=start,
                        end=end,
                        confidence=ent_data.get("confidence", 0.7),
                        context=context,
                        metadata={"openai_extraction": True},
                        extraction_method="openai"
                    )
                    entities.append(entity)
                
                logger.info(f"OpenAI extracted {len(entities)} entities")
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse OpenAI response as JSON")
                
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
        
        return entities
    
    def _extract_with_regex(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []
        
        try:
            for entity_type, patterns in self.legal_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        # Get context
                        start = match.start()
                        end = match.end()
                        context_start = max(0, start - 50)
                        context_end = min(len(text), end + 50)
                        context = text[context_start:context_end]
                        
                        # Map regex entity types to standard types
                        label = self._map_regex_to_standard_label(entity_type)
                        
                        entity = Entity(
                            text=match.group(),
                            label=label,
                            start=start,
                            end=end,
                            confidence=0.9,  # High confidence for regex matches
                            context=context,
                            metadata={
                                "regex_pattern": pattern,
                                "regex_type": entity_type
                            },
                            extraction_method="regex"
                        )
                        entities.append(entity)
            
            logger.info(f"Regex extracted {len(entities)} entities")
            
        except Exception as e:
            logger.error(f"Regex extraction failed: {e}")
        
        return entities
    
    def _map_regex_to_standard_label(self, regex_type: str) -> str:
        """Map regex entity types to standard NER labels."""
        mapping = {
            "case_citations": "CASE",
            "statutes": "STATUTE",
            "dates": "DATE",
            "money": "MONEY",
            "percentages": "PERCENT",
            "party_roles": "PARTY_ROLE",
            "legal_documents": "DOCUMENT_TYPE"
        }
        return mapping.get(regex_type, "LEGAL_CONCEPT")
    
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge overlapping entities from different extraction methods."""
        if not entities:
            return entities
        
        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda x: x.start)
        merged = []
        
        for entity in sorted_entities:
            # Check for overlap with existing entities
            overlapping = False
            
            for i, existing in enumerate(merged):
                if self._entities_overlap(entity, existing):
                    # Merge entities - keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        merged[i] = entity
                    overlapping = True
                    break
            
            if not overlapping:
                merged.append(entity)
        
        return merged
    
    def _entities_overlap(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities overlap."""
        return not (entity1.end <= entity2.start or entity2.end <= entity1.start)
    
    def _validate_entities(self, entities: List[Entity], text: str) -> List[Entity]:
        """Validate extracted entities using validation rules."""
        validated = []
        
        for entity in entities:
            if self._is_valid_entity(entity, text):
                validated.append(entity)
        
        return validated
    
    def _is_valid_entity(self, entity: Entity, text: str) -> bool:
        """Check if an entity meets validation criteria."""
        rules = self.validation_rules
        
        # Check confidence threshold
        if entity.confidence < rules["min_confidence"]:
            return False
        
        # Check length
        if len(entity.text) < rules["min_entity_length"] or len(entity.text) > rules["max_entity_length"]:
            return False
        
        # Check blacklist
        if entity.text.lower() in rules["blacklist_words"]:
            return False
        
        # Check if entity is just whitespace or punctuation
        if not entity.text.strip() or entity.text.isspace():
            return False
        
        return True

    def _extract_relationships(self, entities: List[Entity], text: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []

        try:
            # Simple relationship extraction based on proximity and patterns
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities[i+1:], i+1):
                    # Check if entities are close to each other
                    distance = entity2.start - entity1.end

                    if 0 <= distance <= 100:  # Within 100 characters
                        relationship = self._determine_relationship(entity1, entity2, text)

                        if relationship:
                            relationships.append({
                                "entity1": entity1.text,
                                "entity1_type": entity1.label,
                                "entity2": entity2.text,
                                "entity2_type": entity2.label,
                                "relationship": relationship,
                                "confidence": 0.7,
                                "context": text[entity1.start:entity2.end]
                            })

        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")

        return relationships

    def _determine_relationship(self, entity1: Entity, entity2: Entity, text: str) -> Optional[str]:
        """Determine the relationship between two entities."""
        # Get text between entities
        between_text = text[entity1.end:entity2.start].lower()

        # Define relationship patterns
        relationship_patterns = {
            "represents": ["represents", "attorney for", "counsel for"],
            "vs": ["v.", "vs.", "versus", "against"],
            "employed_by": ["employed by", "works for", "employee of"],
            "located_in": ["located in", "situated in", "based in"],
            "decided_by": ["decided by", "ruled by", "held by"],
            "cited_in": ["cited in", "referenced in", "mentioned in"],
            "party_to": ["party to", "signatory to", "bound by"]
        }

        for relationship, patterns in relationship_patterns.items():
            if any(pattern in between_text for pattern in patterns):
                return relationship

        # Type-based relationships
        if entity1.label == "PERSON" and entity2.label == "ORGANIZATION":
            return "affiliated_with"
        elif entity1.label == "CASE" and entity2.label == "COURT":
            return "decided_by"
        elif entity1.label == "PERSON" and entity2.label == "CASE":
            return "involved_in"

        return None

    def _calculate_entity_counts(self, entities: List[Entity]) -> Dict[str, int]:
        """Calculate counts for each entity type."""
        counts = {}
        for entity in entities:
            counts[entity.label] = counts.get(entity.label, 0) + 1
        return counts

    def _calculate_overall_confidence(self, entities: List[Entity]) -> float:
        """Calculate overall confidence score for the extraction."""
        if not entities:
            return 0.0

        total_confidence = sum(entity.confidence for entity in entities)
        return total_confidence / len(entities)

    def get_entity_visualization(self, text: str, entities: List[Entity]) -> str:
        """
        Generate HTML visualization of entities.

        Args:
            text (str): Original text
            entities (List[Entity]): Extracted entities

        Returns:
            str: HTML visualization
        """
        try:
            if not self.nlp:
                return "spaCy model not available for visualization"

            # Create spaCy doc with custom entities
            doc = self.nlp(text)

            # Convert our entities to spaCy format
            spacy_entities = []
            for entity in entities:
                spacy_entities.append((entity.start, entity.end, entity.label))

            # Create new doc with custom entities
            doc.ents = [doc.char_span(start, end, label=label) for start, end, label in spacy_entities if doc.char_span(start, end)]

            # Generate visualization
            html = displacy.render(doc, style="ent", jupyter=False)
            return html

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return f"Visualization failed: {str(e)}"

    def export_entities_to_json(self, result: NERResult) -> str:
        """
        Export NER results to JSON format.

        Args:
            result (NERResult): NER extraction results

        Returns:
            str: JSON representation
        """
        try:
            export_data = {
                "entities": [
                    {
                        "text": entity.text,
                        "label": entity.label,
                        "start": entity.start,
                        "end": entity.end,
                        "confidence": entity.confidence,
                        "context": entity.context,
                        "extraction_method": entity.extraction_method,
                        "metadata": entity.metadata
                    }
                    for entity in result.entities
                ],
                "entity_counts": result.entity_counts,
                "relationships": result.relationships,
                "confidence_score": result.confidence_score,
                "processing_stats": result.processing_stats,
                "export_timestamp": datetime.now().isoformat()
            }

            return json.dumps(export_data, indent=2)

        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return json.dumps({"error": str(e)})

    def get_entities_by_type(self, result: NERResult, entity_type: str) -> List[Entity]:
        """
        Get all entities of a specific type.

        Args:
            result (NERResult): NER extraction results
            entity_type (str): Entity type to filter by

        Returns:
            List[Entity]: Entities of the specified type
        """
        return [entity for entity in result.entities if entity.label == entity_type]

    def get_entity_statistics(self, result: NERResult) -> Dict[str, Any]:
        """
        Get detailed statistics about extracted entities.

        Args:
            result (NERResult): NER extraction results

        Returns:
            Dict: Detailed statistics
        """
        if not result.entities:
            return {"total_entities": 0}

        # Calculate statistics
        confidence_scores = [entity.confidence for entity in result.entities]
        entity_lengths = [len(entity.text) for entity in result.entities]

        # Method distribution
        method_counts = {}
        for entity in result.entities:
            method = entity.extraction_method
            method_counts[method] = method_counts.get(method, 0) + 1

        # Most common entities
        entity_texts = [entity.text for entity in result.entities]
        entity_frequency = {}
        for text in entity_texts:
            entity_frequency[text] = entity_frequency.get(text, 0) + 1

        most_common = sorted(entity_frequency.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_entities": len(result.entities),
            "unique_entities": len(set(entity_texts)),
            "entity_types": len(result.entity_counts),
            "confidence_stats": {
                "average": sum(confidence_scores) / len(confidence_scores),
                "min": min(confidence_scores),
                "max": max(confidence_scores)
            },
            "length_stats": {
                "average": sum(entity_lengths) / len(entity_lengths),
                "min": min(entity_lengths),
                "max": max(entity_lengths)
            },
            "extraction_methods": method_counts,
            "most_common_entities": most_common,
            "relationships_found": len(result.relationships)
        }

    def train_custom_model(self,
                          training_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]],
                          model_output_path: str,
                          iterations: int = 30) -> bool:
        """
        Train a custom spaCy NER model with legal training data.

        Args:
            training_data: List of (text, {"entities": [(start, end, label)]}) tuples
            model_output_path: Path to save the trained model
            iterations: Number of training iterations

        Returns:
            bool: Success status
        """
        try:
            import random
            from spacy.training import Example

            if not self.nlp:
                logger.error("Base spaCy model not available for training")
                return False

            # Get the NER component
            ner = self.nlp.get_pipe("ner")

            # Add new entity labels
            for _, annotations in training_data:
                for ent in annotations.get("entities", []):
                    ner.add_label(ent[2])

            # Disable other pipeline components during training
            pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
            unaffected_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]

            # Training loop
            with self.nlp.disable_pipes(*unaffected_pipes):
                optimizer = self.nlp.resume_training()

                for iteration in range(iterations):
                    random.shuffle(training_data)
                    losses = {}

                    # Create training examples
                    examples = []
                    for text, annotations in training_data:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)

                    # Update the model
                    self.nlp.update(examples, drop=0.5, losses=losses)

                    if iteration % 10 == 0:
                        logger.info(f"Training iteration {iteration}, Losses: {losses}")

            # Save the trained model
            self.nlp.to_disk(model_output_path)
            logger.info(f"Custom model saved to {model_output_path}")

            return True

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
