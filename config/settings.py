"""
Configuration module for the sophisticated document annotation tool.
Handles model configurations, API settings, and annotation parameters.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class GeminiModel(Enum):
    """Supported Gemini models with their specifications."""
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_0_FLASH = "gemini-2.0-flash-exp"
    GEMINI_2_5_PRO = "gemini-2.5-pro"


class AnnotationType(Enum):
    """Types of linguistic annotations supported."""
    SOCIO_PRAGMATIC = "socio_pragmatic"
    ACADEMIC_STANCE = "academic_stance"
    RAPPORT_BUILDING = "rapport_building"
    DISCOURSE_MARKERS = "discourse_markers"
    POLITENESS_STRATEGIES = "politeness_strategies"
    HEDGING_BOOSTING = "hedging_boosting"
    EPISTEMIC_STANCE = "epistemic_stance"
    EVALUATIVE_LANGUAGE = "evaluative_language"


@dataclass
class ModelConfig:
    """Configuration for Gemini model parameters."""
    model: GeminiModel = GeminiModel.GEMINI_2_5_FLASH
    temperature: float = 0.1
    max_output_tokens: int = 8000
    top_p: float = 0.95
    top_k: int = 40
    candidate_count: int = 1


@dataclass
class SpacyConfig:
    """Configuration for spaCy NLP pipeline."""
    model_name: str = "en_core_web_lg"
    components: List[str] = field(default_factory=lambda: [
        "tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"
    ])
    custom_components: List[str] = field(default_factory=lambda: [
        "sentiment", "textcat", "entity_ruler"
    ])


@dataclass
class AnnotationConfig:
    """Configuration for annotation processes."""
    batch_size: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0
    include_confidence_scores: bool = True
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True
    annotation_types: List[AnnotationType] = field(default_factory=lambda: [
        AnnotationType.SOCIO_PRAGMATIC
    ])


@dataclass
@dataclass
class Config:
    """Main configuration class."""
    model_config: ModelConfig = field(default_factory=ModelConfig)
    spacy_config: SpacyConfig = field(default_factory=SpacyConfig)
    annotation_config: AnnotationConfig = field(default_factory=AnnotationConfig)
    api_key: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY")
        
        # Only require API key for actual LLM operations, not for testing
        if not self.api_key:
            import warnings
            warnings.warn(
                "No Google API key found. Set GOOGLE_API_KEY environment variable "
                "or provide api_key parameter for LLM operations.",
                UserWarning
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "api_key": "***hidden***",
            "model_config": {
                "model": self.model_config.model.value,
                "temperature": self.model_config.temperature,
                "max_output_tokens": self.model_config.max_output_tokens,
                "top_p": self.model_config.top_p,
                "top_k": self.model_config.top_k,
                "candidate_count": self.model_config.candidate_count,
            },
            "spacy_config": {
                "model_name": self.spacy_config.model_name,
                "components": self.spacy_config.components,
                "custom_components": self.spacy_config.custom_components,
            },
            "annotation_config": {
                "batch_size": self.annotation_config.batch_size,
                "max_retries": self.annotation_config.max_retries,
                "retry_delay": self.annotation_config.retry_delay,
                "include_confidence_scores": self.annotation_config.include_confidence_scores,
                "enable_preprocessing": self.annotation_config.enable_preprocessing,
                "enable_postprocessing": self.annotation_config.enable_postprocessing,
                "annotation_types": [t.value for t in self.annotation_config.annotation_types],
            }
        }
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse the configuration data
        model_config = ModelConfig(
            model=GeminiModel(data.get("model_config", {}).get("model", "gemini-2.5-flash")),
            temperature=data.get("model_config", {}).get("temperature", 0.1),
            max_output_tokens=data.get("model_config", {}).get("max_output_tokens", 8000),
            top_p=data.get("model_config", {}).get("top_p", 0.95),
            top_k=data.get("model_config", {}).get("top_k", 40),
            candidate_count=data.get("model_config", {}).get("candidate_count", 1),
        )
        
        spacy_config = SpacyConfig(
            model_name=data.get("spacy_config", {}).get("model_name", "en_core_web_lg"),
            components=data.get("spacy_config", {}).get("components", []),
            custom_components=data.get("spacy_config", {}).get("custom_components", []),
        )
        
        annotation_types = [
            AnnotationType(t) for t in data.get("annotation_config", {}).get("annotation_types", ["socio_pragmatic"])
        ]
        
        annotation_config = AnnotationConfig(
            batch_size=data.get("annotation_config", {}).get("batch_size", 5),
            max_retries=data.get("annotation_config", {}).get("max_retries", 3),
            retry_delay=data.get("annotation_config", {}).get("retry_delay", 1.0),
            include_confidence_scores=data.get("annotation_config", {}).get("include_confidence_scores", True),
            enable_preprocessing=data.get("annotation_config", {}).get("enable_preprocessing", True),
            enable_postprocessing=data.get("annotation_config", {}).get("enable_postprocessing", True),
            annotation_types=annotation_types,
        )
        
        return cls(
            api_key=data.get("api_key"),
            model_config=model_config,
            spacy_config=spacy_config,
            annotation_config=annotation_config,
        )
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# Default configuration instance
DEFAULT_CONFIG = Config()
