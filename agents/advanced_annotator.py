"""
LangChain-based advanced linguistic annotator with Gemini integration.
Provides sophisticated socio-pragmatic annotation capabilities.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback
from langchain.schema.output_parser import OutputParserException

from config.settings import Config, GeminiModel, AnnotationType
from prompts.advanced_templates import PromptChain, AdvancedPromptTemplates, PromptType
from nlp.spacy_analyzer import AdvancedNLPAnalyzer, AnalysisResult


logger = logging.getLogger(__name__)


class AdvancedAnnotator:
    """Advanced linguistic annotator using LangChain and Gemini models."""
    
    def __init__(self, config: Config, prompt_chain: Optional[PromptChain] = None):
        """
        Initialize the annotator with configuration and prompt chain.
        
        Args:
            config: Configuration object with model and annotation settings
            prompt_chain: Optional prompt chain for multi-step analysis
        """
        self.config = config
        self.prompt_chain = prompt_chain
        self.nlp_analyzer = AdvancedNLPAnalyzer(config.spacy_config)
        self.llm = None  # Initialize lazily when needed
        
        # Initialize statistics tracking
        self.stats = {
            "total_requests": 0,
            "successful_annotations": 0,
            "failed_annotations": 0,
            "total_tokens_used": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
        }
    
    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Get LLM instance, initializing if needed."""
        if self.llm is None:
            self.llm = self._initialize_llm()
        return self.llm
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize the LangChain Gemini LLM."""
        if not self.config.api_key:
            raise ValueError(
                "Google API key is required for LLM operations. "
                "Set GOOGLE_API_KEY environment variable or provide api_key in config."
            )
            
        model_config = self.config.model_config
        
        return ChatGoogleGenerativeAI(
            model=model_config.model.value,
            google_api_key=self.config.api_key,
            temperature=model_config.temperature,
            max_output_tokens=model_config.max_output_tokens,
            top_p=model_config.top_p,
            top_k=model_config.top_k,
            candidate_count=model_config.candidate_count,
        )
    
    def annotate_text(
        self, 
        text: str, 
        text_id: Optional[str] = None,
        annotation_types: Optional[List[AnnotationType]] = None,
        include_nlp_analysis: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Annotate a single text with specified annotation types.
        
        Args:
            text: Text to annotate
            text_id: Optional identifier for the text
            annotation_types: Types of annotations to perform
            include_nlp_analysis: Whether to include spaCy NLP analysis
            **kwargs: Additional parameters for prompt formatting
        
        Returns:
            Dictionary containing annotation results
        """
        start_time = time.time()
        
        if text_id is None:
            text_id = f"text_{int(time.time())}"
        
        if annotation_types is None:
            annotation_types = self.config.annotation_config.annotation_types
        
        try:
            result = {
                "text_id": text_id,
                "original_text": text,
                "llm_annotations": {},
                "nlp_analysis": None,
                "combined_features": [],
                "metadata": {
                    "annotation_types": [t.value for t in annotation_types],
                    "model_used": self.config.model_config.model.value,
                    "timestamp": time.time(),
                }
            }
            
            # Perform spaCy analysis if requested
            if include_nlp_analysis:
                nlp_result = self.nlp_analyzer.analyze_text(text)
                result["nlp_analysis"] = asdict(nlp_result)
                result["combined_features"].extend(nlp_result.features)
            
            # Perform LLM-based annotation
            if self.prompt_chain:
                llm_annotations = self._annotate_with_chain(text, text_id, **kwargs)
            else:
                llm_annotations = self._annotate_with_individual_prompts(
                    text, text_id, annotation_types, **kwargs
                )
            
            result["llm_annotations"] = llm_annotations
            
            # Combine and reconcile features
            if include_nlp_analysis and llm_annotations:
                result["combined_features"] = self._combine_features(
                    result["nlp_analysis"]["features"], llm_annotations
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            result["metadata"]["processing_time"] = processing_time
            
            # Update statistics
            self.stats["successful_annotations"] += 1
            self.stats["average_response_time"] = (
                (self.stats["average_response_time"] * (self.stats["successful_annotations"] - 1) + processing_time) /
                self.stats["successful_annotations"]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error annotating text {text_id}: {str(e)}")
            self.stats["failed_annotations"] += 1
            
            return {
                "text_id": text_id,
                "original_text": text,
                "error": str(e),
                "metadata": {
                    "annotation_types": [t.value for t in annotation_types] if annotation_types else [],
                    "model_used": self.config.model_config.model.value,
                    "timestamp": time.time(),
                    "processing_time": time.time() - start_time,
                }
            }
    
    def _annotate_with_chain(self, text: str, text_id: str, **kwargs) -> Dict[str, Any]:
        """Annotate using a predefined prompt chain."""
        template = self.prompt_chain.get_combined_template()
        
        # Format the prompt
        formatted_messages = template.format_messages(
            text=text,
            text_id=text_id,
            **kwargs
        )
        
        # Get LLM response
        response = self._call_llm_with_retry(formatted_messages)
        
        # Parse response
        return self._parse_llm_response(response.content)
    
    def _annotate_with_individual_prompts(
        self, 
        text: str, 
        text_id: str, 
        annotation_types: List[AnnotationType],
        **kwargs
    ) -> Dict[str, Any]:
        """Annotate using individual prompts for each annotation type."""
        results = {}
        
        # Map annotation types to prompt types
        type_mapping = {
            AnnotationType.SOCIO_PRAGMATIC: PromptType.SOCIO_PRAGMATIC,
            AnnotationType.ACADEMIC_STANCE: PromptType.ACADEMIC_STANCE,
            AnnotationType.RAPPORT_BUILDING: PromptType.RAPPORT_BUILDING,
            AnnotationType.POLITENESS_STRATEGIES: PromptType.POLITENESS_STRATEGIES,
            AnnotationType.HEDGING_BOOSTING: PromptType.HEDGING_BOOSTING,
        }
        
        for annotation_type in annotation_types:
            if annotation_type in type_mapping:
                try:
                    prompt_type = type_mapping[annotation_type]
                    template = AdvancedPromptTemplates.get_prompt_template(prompt_type)
                    
                    formatted_messages = template.format_messages(
                        text=text,
                        text_id=text_id,
                        **kwargs
                    )
                    
                    response = self._call_llm_with_retry(formatted_messages)
                    parsed_response = self._parse_llm_response(response.content)
                    
                    results[annotation_type.value] = parsed_response
                    
                except Exception as e:
                    logger.error(f"Error in {annotation_type.value} annotation: {str(e)}")
                    results[annotation_type.value] = {"error": str(e)}
        
        return results
    
    def _call_llm_with_retry(self, messages: List[BaseMessage]) -> BaseMessage:
        """Call LLM with retry logic and statistics tracking."""
        max_retries = self.config.annotation_config.max_retries
        retry_delay = self.config.annotation_config.retry_delay
        
        for attempt in range(max_retries + 1):
            try:
                self.stats["total_requests"] += 1
                
                # Track token usage if available
                with get_openai_callback() as cb:
                    response = self._get_llm()(messages)
                    if hasattr(cb, 'total_tokens'):
                        self.stats["total_tokens_used"] += cb.total_tokens
                    if hasattr(cb, 'total_cost'):
                        self.stats["total_cost"] += cb.total_cost
                
                return response
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"LLM call failed after {max_retries + 1} attempts: {str(e)}")
                    raise e
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response, handling various JSON formats."""
        # Clean up the response text
        cleaned_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        cleaned_text = cleaned_text.strip()
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {str(e)}")
            logger.warning(f"Raw response: {response_text[:500]}...")
            
            # Attempt to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Return the raw text if JSON parsing fails
            return {
                "raw_response": response_text,
                "parsing_error": str(e),
                "parsed_successfully": False
            }
    
    def _combine_features(self, nlp_features: List[Dict], llm_annotations: Dict[str, Any]) -> List[Dict]:
        """Combine and reconcile features from NLP analysis and LLM annotations."""
        combined = nlp_features.copy() if nlp_features else []
        
        # Extract features from LLM annotations
        for annotation_type, annotation_data in llm_annotations.items():
            if isinstance(annotation_data, dict) and not annotation_data.get("error"):
                # Look for feature lists in the annotation data
                for key, value in annotation_data.items():
                    if isinstance(value, list) and key.endswith(('_features', '_markers', '_strategies')):
                        for feature in value:
                            if isinstance(feature, dict) and 'span' in feature:
                                combined_feature = {
                                    "feature_type": f"llm_{annotation_type}_{key}",
                                    "text": feature.get("span", ""),
                                    "start_char": feature.get("start_char", -1),
                                    "end_char": feature.get("end_char", -1),
                                    "confidence": feature.get("confidence", 1.0),
                                    "source": "llm",
                                    "annotation_type": annotation_type,
                                    "properties": feature
                                }
                                combined.append(combined_feature)
        
        # Remove duplicates and resolve conflicts
        return self._deduplicate_features(combined)
    
    def _deduplicate_features(self, features: List[Dict]) -> List[Dict]:
        """Remove duplicate features and resolve conflicts."""
        # Simple deduplication based on text spans
        seen_spans = set()
        deduplicated = []
        
        for feature in features:
            span_key = (feature.get("text", ""), feature.get("start_char", -1), feature.get("end_char", -1))
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                deduplicated.append(feature)
        
        return deduplicated
    
    def annotate_batch(
        self, 
        texts: List[str], 
        text_ids: Optional[List[str]] = None,
        annotation_types: Optional[List[AnnotationType]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Annotate a batch of texts.
        
        Args:
            texts: List of texts to annotate
            text_ids: Optional list of text identifiers
            annotation_types: Types of annotations to perform
            **kwargs: Additional parameters for prompt formatting
        
        Returns:
            List of annotation results
        """
        if text_ids is None:
            text_ids = [f"text_{i}" for i in range(len(texts))]
        
        if len(text_ids) != len(texts):
            raise ValueError("Number of text_ids must match number of texts")
        
        results = []
        for text, text_id in zip(texts, text_ids):
            result = self.annotate_text(
                text=text,
                text_id=text_id,
                annotation_types=annotation_types,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get annotation statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset annotation statistics."""
        self.stats = {
            "total_requests": 0,
            "successful_annotations": 0,
            "failed_annotations": 0,
            "total_tokens_used": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
        }
    
    def switch_model(self, new_model: GeminiModel):
        """Switch to a different Gemini model."""
        self.config.model_config.model = new_model
        self.llm = None  # Reset to force re-initialization with new model
        logger.info(f"Switched to model: {new_model.value}")
    
    def update_config(self, new_config: Config):
        """Update the configuration and reinitialize components as needed."""
        old_model = self.config.model_config.model
        self.config = new_config
        
        # Reinitialize LLM if model changed
        if new_config.model_config.model != old_model:
            self.llm = None  # Reset to force re-initialization
        
        # Reinitialize NLP analyzer if spacy config changed
        self.nlp_analyzer = AdvancedNLPAnalyzer(new_config.spacy_config)
        
        logger.info("Configuration updated successfully")
