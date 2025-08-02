"""
Advanced Document Annotation Tool

A sophisticated socio-pragmatic analysis system using LangChain, Gemini models, and spaCy
for comprehensive linguistic annotation of academic and professional texts.

Key Features:
- LangChain integration for robust prompt management
- Multiple Gemini model support (2.5-Flash, 2.0-Flash, 2.5-Pro)
- Advanced spaCy NLP analysis
- Socio-pragmatic feature detection
- Academic discourse analysis
- Politeness and stance detection

Example Usage:
    from config import Config
    from agents import AdvancedAnnotator
    
    config = Config(api_key="your_api_key")
    annotator = AdvancedAnnotator(config)
    
    result = annotator.annotate_text("Your text here")
"""

__version__ = "2.0.0"
__author__ = "Advanced Annotation Tool Team"
__description__ = "Sophisticated document annotation with socio-pragmatic analysis"

# Import main components for easy access
from config import Config, GeminiModel, AnnotationType
from agents import AdvancedAnnotator
from data import AdvancedDataLoader
from prompts import PromptChain, PromptType
from generators import SyntheticDataGenerator, GenerationParameters, TextGenreType

__all__ = [
    'Config',
    'GeminiModel', 
    'AnnotationType',
    'AdvancedAnnotator',
    'AdvancedDataLoader',
    'PromptChain',
    'PromptType',
    'SyntheticDataGenerator',
    'GenerationParameters',
    'TextGenreType'
]
