"""
Synthetic data generation module for creating natural, human-like texts.
Uses LLM prompts combined with spaCy analysis to generate realistic data points.
"""

import json
import logging
import time
import random
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage

from config.settings import Config, GeminiModel, AnnotationType
from nlp.spacy_analyzer import AdvancedNLPAnalyzer


logger = logging.getLogger(__name__)


class TextGenreType(Enum):
    """Types of text genres for synthetic generation."""
    ACADEMIC_PAPER = "academic_paper"
    THESIS_CHAPTER = "thesis_chapter"
    ESSAY = "essay"
    BUSINESS_EMAIL = "business_email"
    GRANT_PROPOSAL = "grant_proposal"
    POLICY_DOCUMENT = "policy_document"
    RESEARCH_REPORT = "research_report"
    CONFERENCE_ABSTRACT = "conference_abstract"
    PEER_REVIEW = "peer_review"
    BLOG_POST = "blog_post"


class LinguisticComplexity(Enum):
    """Complexity levels for generated texts."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class GenerationParameters:
    """Parameters for controlling text generation."""
    genre: TextGenreType = TextGenreType.ACADEMIC_PAPER
    complexity: LinguisticComplexity = LinguisticComplexity.MODERATE
    target_features: List[str] = field(default_factory=list)
    word_count_range: tuple = (100, 300)
    topic_area: Optional[str] = None
    stance_type: Optional[str] = None  # confident, tentative, neutral
    politeness_level: Optional[str] = None  # formal, informal, mixed
    target_readability: Optional[float] = None  # Flesch reading ease score
    include_citations: bool = False
    author_persona: Optional[str] = None  # expert, student, practitioner
    audience_level: Optional[str] = None  # general, academic, specialist


@dataclass
class GeneratedText:
    """Result of text generation with metadata."""
    text: str
    generation_id: str
    parameters: GenerationParameters
    actual_features: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SyntheticDataGenerator:
    """Generator for creating synthetic texts with specific linguistic features."""
    
    def __init__(self, config: Config):
        """
        Initialize the synthetic data generator.
        
        Args:
            config: Configuration object with model settings
        """
        self.config = config
        self.llm = None  # Initialize lazily when needed
        self.nlp_analyzer = AdvancedNLPAnalyzer(config.spacy_config)
        
        # Generation statistics
        self.stats = {
            "total_generated": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "average_generation_time": 0.0,
        }
        
        # Template cache for efficiency
        self._template_cache = {}
    
    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Get LLM instance, initializing if needed."""
        if self.llm is None:
            self.llm = self._initialize_llm()
        return self.llm

    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize the LangChain Gemini LLM for generation."""
        if not self.config.api_key:
            raise ValueError(
                "Google API key is required for text generation. "
                "Set GOOGLE_API_KEY environment variable or provide api_key in config."
            )
            
        model_config = self.config.model_config
        
        return ChatGoogleGenerativeAI(
            model=model_config.model.value,
            google_api_key=self.config.api_key,
            temperature=0.7,  # Higher temperature for more creative generation
            max_output_tokens=model_config.max_output_tokens,
            top_p=model_config.top_p,
            top_k=model_config.top_k,
        )
    
    def _create_generation_parameters(
        self,
        genre: TextGenreType,
        topic: Optional[str] = None,
        complexity: LinguisticComplexity = LinguisticComplexity.MODERATE,
        target_features: Optional[List[str]] = None,
        word_count_range: tuple = (100, 300)
    ) -> GenerationParameters:
        """Create generation parameters for synthetic text generation."""
        return GenerationParameters(
            genre=genre,
            complexity=complexity,
            target_features=target_features or [],
            word_count_range=word_count_range,
            topic_area=topic,
            stance_type="neutral",
            politeness_level="formal",
            target_readability=None,
            include_citations=False,
            author_persona="researcher",
            audience_level="academic"
        )
    
    def generate_text(
        self,
        parameters: GenerationParameters,
        generation_id: Optional[str] = None
    ) -> GeneratedText:
        """
        Generate a synthetic text with specified parameters.
        
        Args:
            parameters: Generation parameters
            generation_id: Optional identifier for the generated text
        
        Returns:
            GeneratedText object with the result
        """
        start_time = time.time()
        
        if generation_id is None:
            generation_id = f"gen_{int(time.time())}_{random.randint(1000, 9999)}"
        
        try:
            # Generate the text using LLM
            generated_text = self._generate_with_llm(parameters)
            
            # Analyze the generated text
            analysis = self.nlp_analyzer.analyze_text(generated_text)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(generated_text, parameters, analysis)
            
            # Extract actual features
            actual_features = self._extract_actual_features(analysis)
            
            generation_time = time.time() - start_time
            
            result = GeneratedText(
                text=generated_text,
                generation_id=generation_id,
                parameters=parameters,
                actual_features=actual_features,
                quality_metrics=quality_metrics,
                generation_time=generation_time,
                metadata={
                    "word_count": len(generated_text.split()),
                    "sentence_count": analysis.discourse_structure.get("sentence_count", 0),
                    "model_used": self.config.model_config.model.value,
                }
            )
            
            self.stats["successful_generations"] += 1
            self.stats["total_generated"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating text {generation_id}: {str(e)}")
            self.stats["failed_generations"] += 1
            self.stats["total_generated"] += 1
            
            return GeneratedText(
                text="",
                generation_id=generation_id,
                parameters=parameters,
                metadata={"error": str(e), "generation_time": time.time() - start_time}
            )
    
    def _generate_with_llm(self, parameters: GenerationParameters) -> str:
        """Generate text using the LLM with appropriate prompts."""
        # Get or create appropriate prompt template
        template = self._get_generation_template(parameters)
        
        # Format the prompt with parameters
        messages = template.format_messages(
            genre=parameters.genre.value,
            complexity=parameters.complexity.value,
            target_features=", ".join(parameters.target_features) if parameters.target_features else "natural academic discourse",
            word_count_min=parameters.word_count_range[0],
            word_count_max=parameters.word_count_range[1],
            topic_area=parameters.topic_area or "general academic topic",
            stance_type=parameters.stance_type or "balanced",
            politeness_level=parameters.politeness_level or "formal academic",
            include_citations="include citations" if parameters.include_citations else "no citations needed",
            author_persona=parameters.author_persona or "academic researcher",
            audience_level=parameters.audience_level or "academic"
        )
        
        # Generate response
        response = self._get_llm()(messages)
        
        # Clean and extract the generated text
        return self._clean_generated_text(response.content)
    
    def _get_generation_template(self, parameters: GenerationParameters) -> ChatPromptTemplate:
        """Get or create a prompt template for the given parameters."""
        template_key = f"{parameters.genre.value}_{parameters.complexity.value}"
        
        if template_key not in self._template_cache:
            self._template_cache[template_key] = self._create_generation_template(parameters)
        
        return self._template_cache[template_key]
    
    def _create_generation_template(self, parameters: GenerationParameters) -> ChatPromptTemplate:
        """Create a specialized prompt template for text generation."""
        
        system_prompt = """You are an expert writer and linguist capable of generating natural, human-like texts across various genres and styles. Your task is to create realistic synthetic texts that demonstrate specific linguistic features while maintaining natural flow and coherence.

Key principles:
1. Generate texts that sound authentically human-written
2. Incorporate specified linguistic features naturally
3. Match the requested genre conventions and style
4. Maintain appropriate complexity and readability
5. Include realistic content and coherent argumentation
6. Vary sentence structures and vocabulary naturally

You excel at creating texts for research purposes, data augmentation, and linguistic analysis."""

        human_prompt = """Generate a {genre} text with the following specifications:

**Content Requirements:**
- Genre: {genre}
- Topic area: {topic_area}
- Target word count: {word_count_min}-{word_count_max} words
- Author persona: {author_persona}
- Target audience: {audience_level}

**Linguistic Features:**
- Complexity level: {complexity}
- Target linguistic features: {target_features}
- Stance type: {stance_type}
- Politeness level: {politeness_level}
- Citation requirements: {include_citations}

**Style Guidelines:**
- Write in a natural, human-like style
- Incorporate the specified linguistic features organically
- Maintain genre-appropriate conventions
- Use varied sentence structures and vocabulary
- Ensure coherent flow and logical organization

**Output Instructions:**
- Provide only the generated text
- Do not include meta-commentary or explanations
- Ensure the text demonstrates the requested features naturally
- Make it sound like authentic human writing in the specified genre

Generate the text now:"""

        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message = HumanMessagePromptTemplate.from_template(human_prompt)
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def _clean_generated_text(self, raw_text: str) -> str:
        """Clean and format the generated text."""
        # Remove any markdown formatting
        text = raw_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split('\n')
            text = '\n'.join(lines[1:-1]) if len(lines) > 2 else text
        
        # Clean up excessive whitespace
        import re
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple line breaks
        text = re.sub(r' +', ' ', text)  # Multiple spaces
        
        return text.strip()
    
    def _calculate_quality_metrics(
        self, 
        text: str, 
        parameters: GenerationParameters, 
        analysis
    ) -> Dict[str, float]:
        """Calculate quality metrics for the generated text."""
        metrics = {}
        
        # Word count accuracy
        actual_word_count = len(text.split())
        target_min, target_max = parameters.word_count_range
        if target_min <= actual_word_count <= target_max:
            metrics["word_count_accuracy"] = 1.0
        else:
            # Calculate how far off we are
            if actual_word_count < target_min:
                metrics["word_count_accuracy"] = actual_word_count / target_min
            else:
                metrics["word_count_accuracy"] = target_max / actual_word_count
        
        # Readability score
        if hasattr(analysis, 'readability_scores') and analysis.readability_scores:
            flesch_score = analysis.readability_scores.get('flesch_reading_ease', 50)
            metrics["readability_score"] = flesch_score
            
            # Readability appropriateness for complexity level
            target_ranges = {
                LinguisticComplexity.SIMPLE: (60, 100),
                LinguisticComplexity.MODERATE: (30, 60),
                LinguisticComplexity.COMPLEX: (0, 30),
                LinguisticComplexity.EXPERT: (0, 20)
            }
            
            target_range = target_ranges.get(parameters.complexity, (30, 60))
            if target_range[0] <= flesch_score <= target_range[1]:
                metrics["complexity_appropriateness"] = 1.0
            else:
                # Calculate deviation
                if flesch_score < target_range[0]:
                    metrics["complexity_appropriateness"] = flesch_score / target_range[0]
                else:
                    metrics["complexity_appropriateness"] = target_range[1] / flesch_score
        
        # Feature presence (if target features specified)
        if parameters.target_features:
            feature_count = len([f for f in analysis.features if any(target in f.feature_type for target in parameters.target_features)])
            metrics["target_feature_density"] = feature_count / len(text.split()) * 100
        
        # Overall quality score
        quality_scores = [v for v in metrics.values() if isinstance(v, float) and 0 <= v <= 1]
        if quality_scores:
            metrics["overall_quality"] = sum(quality_scores) / len(quality_scores)
        
        return metrics
    
    def _extract_actual_features(self, analysis) -> Dict[str, Any]:
        """Extract actual linguistic features from the analysis."""
        features = {
            "total_features": len(analysis.features),
            "feature_types": {},
            "readability": analysis.readability_scores,
            "discourse_structure": analysis.discourse_structure
        }
        
        # Count features by type
        for feature in analysis.features:
            feature_type = feature.feature_type
            features["feature_types"][feature_type] = features["feature_types"].get(feature_type, 0) + 1
        
        return features
    
    def generate_batch(
        self,
        parameters_list: List[GenerationParameters],
        batch_id: Optional[str] = None
    ) -> List[GeneratedText]:
        """
        Generate multiple texts in batch.
        
        Args:
            parameters_list: List of generation parameters
            batch_id: Optional batch identifier
        
        Returns:
            List of GeneratedText objects
        """
        if batch_id is None:
            batch_id = f"batch_{int(time.time())}"
        
        results = []
        for i, params in enumerate(parameters_list):
            generation_id = f"{batch_id}_{i}"
            result = self.generate_text(params, generation_id)
            results.append(result)
        
        return results
    
    def generate_dataset(
        self,
        genre: TextGenreType,
        count: int,
        variety_parameters: Optional[Dict[str, List[Any]]] = None,
        save_path: Optional[str] = None
    ) -> List[GeneratedText]:
        """
        Generate a diverse dataset of texts.
        
        Args:
            genre: Base genre for all texts
            count: Number of texts to generate
            variety_parameters: Parameters to vary across generations
            save_path: Optional path to save the dataset
        
        Returns:
            List of generated texts
        """
        if variety_parameters is None:
            variety_parameters = {
                "complexity": list(LinguisticComplexity),
                "stance_type": ["confident", "tentative", "neutral"],
                "politeness_level": ["formal", "informal", "mixed"],
                "word_count_range": [(50, 150), (150, 250), (250, 400)]
            }
        
        results = []
        dataset_id = f"dataset_{genre.value}_{int(time.time())}"
        
        for i in range(count):
            # Create varied parameters
            params = GenerationParameters(genre=genre)
            
            # Randomize variety parameters
            for param_name, param_values in variety_parameters.items():
                if hasattr(params, param_name):
                    setattr(params, param_name, random.choice(param_values))
            
            generation_id = f"{dataset_id}_{i}"
            result = self.generate_text(params, generation_id)
            results.append(result)
        
        # Save dataset if path provided
        if save_path:
            self.save_dataset(results, save_path)
        
        return results
    
    def save_dataset(self, generated_texts: List[GeneratedText], file_path: str):
        """Save generated dataset to file."""
        dataset = []
        for text_obj in generated_texts:
            dataset.append({
                "generation_id": text_obj.generation_id,
                "text": text_obj.text,
                "parameters": {
                    "genre": text_obj.parameters.genre.value,
                    "complexity": text_obj.parameters.complexity.value,
                    "target_features": text_obj.parameters.target_features,
                    "word_count_range": text_obj.parameters.word_count_range,
                    "topic_area": text_obj.parameters.topic_area,
                    "stance_type": text_obj.parameters.stance_type,
                    "politeness_level": text_obj.parameters.politeness_level,
                },
                "actual_features": text_obj.actual_features,
                "quality_metrics": text_obj.quality_metrics,
                "metadata": text_obj.metadata
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved dataset with {len(dataset)} texts to {file_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        if self.stats["successful_generations"] > 0:
            self.stats["average_generation_time"] = (
                self.stats.get("total_generation_time", 0) / self.stats["successful_generations"]
            )
        
        return self.stats.copy()


# Convenience functions for common use cases
def generate_academic_texts(
    config: Config,
    count: int = 10,
    complexity: LinguisticComplexity = LinguisticComplexity.MODERATE
) -> List[GeneratedText]:
    """Generate academic texts with varied features."""
    generator = SyntheticDataGenerator(config)
    
    variety_params = {
        "complexity": [complexity],
        "target_features": [
            ["hedging", "academic_stance"],
            ["boosting", "evaluative_stance"],
            ["rapport_building", "politeness"],
            ["epistemic_stance", "citations"]
        ],
        "stance_type": ["confident", "tentative", "balanced"],
        "include_citations": [True, False]
    }
    
    return generator.generate_dataset(
        TextGenreType.ACADEMIC_PAPER,
        count,
        variety_params
    )


def generate_business_emails(
    config: Config,
    count: int = 10,
    politeness_focus: bool = True
) -> List[GeneratedText]:
    """Generate business emails with politeness variations."""
    generator = SyntheticDataGenerator(config)
    
    if politeness_focus:
        variety_params = {
            "politeness_level": ["formal", "informal", "mixed"],
            "target_features": [
                ["negative_politeness", "rapport_building"],
                ["positive_politeness", "directness"],
                ["hedging", "deference"]
            ],
            "complexity": [LinguisticComplexity.SIMPLE, LinguisticComplexity.MODERATE]
        }
    else:
        variety_params = {
            "complexity": list(LinguisticComplexity),
            "stance_type": ["confident", "neutral", "collaborative"]
        }
    
    return generator.generate_dataset(
        TextGenreType.BUSINESS_EMAIL,
        count,
        variety_params
    )
