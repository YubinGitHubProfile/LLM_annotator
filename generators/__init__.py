"""
Synthetic data generation module for creating realistic texts with specific linguistic features.
"""

from .synthetic_data import (
    SyntheticDataGenerator,
    GenerationParameters,
    GeneratedText,
    TextGenreType,
    LinguisticComplexity,
    generate_academic_texts,
    generate_business_emails
)

__all__ = [
    'SyntheticDataGenerator',
    'GenerationParameters', 
    'GeneratedText',
    'TextGenreType',
    'LinguisticComplexity',
    'generate_academic_texts',
    'generate_business_emails'
]
