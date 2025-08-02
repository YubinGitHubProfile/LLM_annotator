"""
Example usage of the advanced document annotation tool.
Demonstrates sophisticated socio-pragmatic analysis capabilities.
"""

import os
import json
from pathlib import Path

# Import the advanced annotation components
from config.settings import Config, GeminiModel, AnnotationType
from agents.advanced_annotator import AdvancedAnnotator
from data.advanced_loader import AdvancedDataLoader
from prompts.advanced_templates import PromptChain, PromptType, COMPREHENSIVE_ANALYSIS_CHAIN
from generators.synthetic_data import (
    SyntheticDataGenerator, 
    GenerationParameters, 
    TextGenreType, 
    LinguisticComplexity,
    generate_academic_texts
)


def setup_environment():
    """Setup the environment and configuration."""
    # Set up API key (replace with your actual key or set as environment variable)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Please set your GOOGLE_API_KEY environment variable")
        return None
    
    # Create configuration
    config = Config(
        api_key=api_key,
    )
    
    # Customize model settings for academic analysis
    config.model_config.model = GeminiModel.GEMINI_2_5_FLASH  # Default high-performance model
    config.model_config.temperature = 0.1  # Low temperature for consistency
    config.model_config.max_output_tokens = 8000  # Increased for detailed analysis
    
    # Configure annotation types for comprehensive analysis
    config.annotation_config.annotation_types = [
        AnnotationType.SOCIO_PRAGMATIC,
        AnnotationType.ACADEMIC_STANCE,
        AnnotationType.RAPPORT_BUILDING,
        AnnotationType.HEDGING_BOOSTING,
        AnnotationType.POLITENESS_STRATEGIES
    ]
    
    # Set batch size for processing
    config.annotation_config.batch_size = 3  # Smaller batches for complex analysis
    
    return config


def example_single_text_annotation():
    """Example of annotating a single academic text."""
    print("=== Single Text Annotation Example ===")
    
    config = setup_environment()
    if not config:
        return
    
    # Create annotator with comprehensive analysis chain
    annotator = AdvancedAnnotator(config, prompt_chain=COMPREHENSIVE_ANALYSIS_CHAIN)
    
    # Example academic text with various socio-pragmatic features
    academic_text = """
    This study presents a novel approach to understanding discourse markers in academic writing. 
    We argue that previous research has inadequately addressed the complexity of these phenomena. 
    Our findings suggest that hedging strategies serve multiple functions beyond mere politeness.
    It is clear that further investigation is warranted. However, one might question whether 
    the methodology employed here is sufficiently robust. Nevertheless, the evidence we present 
    demonstrates the importance of this line of inquiry. In our view, this work contributes 
    significantly to the field and opens new avenues for research.
    """
    
    # Perform annotation
    print("Analyzing text...")
    result = annotator.annotate_text(
        text=academic_text,
        text_id="academic_sample_1",
        include_nlp_analysis=True
    )
    
    # Display results
    print(f"\nText ID: {result['text_id']}")
    print(f"Processing Time: {result['metadata']['processing_time']:.2f} seconds")
    
    # Show LLM annotations
    if 'llm_annotations' in result:
        print("\n--- LLM Annotations ---")
        for annotation_type, annotations in result['llm_annotations'].items():
            print(f"\n{annotation_type.upper()}:")
            if isinstance(annotations, dict) and not annotations.get('error'):
                print(json.dumps(annotations, indent=2)[:500] + "...")
            else:
                print(f"  Error or unexpected format: {annotations}")
    
    # Show spaCy analysis summary
    if 'nlp_analysis' in result and result['nlp_analysis']:
        nlp_stats = {
            'total_features': len(result['nlp_analysis']['features']),
            'readability_scores': result['nlp_analysis']['readability_scores'],
            'discourse_structure': result['nlp_analysis']['discourse_structure']
        }
        print(f"\n--- spaCy Analysis Summary ---")
        print(json.dumps(nlp_stats, indent=2))
    
    # Show combined features summary
    feature_types = {}
    for feature in result.get('combined_features', []):
        feature_type = feature.get('feature_type', 'unknown')
        feature_types[feature_type] = feature_types.get(feature_type, 0) + 1
    
    print(f"\n--- Combined Features Summary ---")
    print(f"Total features detected: {len(result.get('combined_features', []))}")
    for feature_type, count in sorted(feature_types.items()):
        print(f"  {feature_type}: {count}")
    
    # Show annotator statistics
    stats = annotator.get_statistics()
    print(f"\n--- Annotator Statistics ---")
    print(json.dumps(stats, indent=2))
    
    return result


def example_csv_annotation():
    """Example of annotating a CSV file with academic texts."""
    print("\n=== CSV File Annotation Example ===")
    
    config = setup_environment()
    if not config:
        return
    
    # Create sample CSV data for demonstration
    sample_data = [
        {
            "id": 1,
            "text": "We believe this research contributes significantly to our understanding of linguistic phenomena.",
            "source": "academic_paper_1"
        },
        {
            "id": 2,
            "text": "It seems that previous studies may have overlooked important factors. However, our analysis suggests otherwise.",
            "source": "academic_paper_2"
        },
        {
            "id": 3,
            "text": "The results clearly demonstrate the effectiveness of this approach. Obviously, further research is needed.",
            "source": "academic_paper_3"
        }
    ]
    
    # Create sample CSV file
    import pandas as pd
    sample_csv = "sample_academic_texts.csv"
    df = pd.DataFrame(sample_data)
    df.to_csv(sample_csv, index=False)
    
    # Create annotator and data loader
    annotator = AdvancedAnnotator(config)
    data_loader = AdvancedDataLoader(config, annotator)
    
    # Annotate the CSV file
    print(f"Annotating CSV file: {sample_csv}")
    summary = data_loader.annotate_csv(
        input_file=sample_csv,
        output_file="annotated_results.json",
        text_column="text",
        id_column="id",
        metadata_columns=["source"],
        annotation_types=[
            AnnotationType.HEDGING_BOOSTING,
            AnnotationType.ACADEMIC_STANCE
        ],
        include_nlp_analysis=True
    )
    
    print(f"\n--- Processing Summary ---")
    print(json.dumps(summary, indent=2, default=str))
    
    # Clean up sample file
    try:
        os.remove(sample_csv)
    except:
        pass


def example_custom_prompt_chain():
    """Example of using custom prompt chains for specific analysis needs."""
    print("\n=== Custom Prompt Chain Example ===")
    
    config = setup_environment()
    if not config:
        return
    
    # Create a custom prompt chain focused on academic stance
    custom_chain = PromptChain([
        PromptType.ACADEMIC_STANCE,
        PromptType.HEDGING_BOOSTING
    ])
    
    annotator = AdvancedAnnotator(config, prompt_chain=custom_chain)
    
    # Example text with strong stance markers
    stance_text = """
    It is undeniable that climate change represents the most pressing challenge of our time.
    However, some researchers might argue that economic factors should take precedence.
    We strongly disagree with this position. The evidence clearly shows that immediate action is required.
    Perhaps critics will contend that our approach is too aggressive, but we believe that
    the urgency of the situation demands bold measures.
    """
    
    result = annotator.annotate_text(
        text=stance_text,
        text_id="stance_analysis_sample",
        include_nlp_analysis=False  # Focus only on LLM analysis
    )
    
    print(f"Custom analysis result for text: {result['text_id']}")
    if 'llm_annotations' in result:
        for annotation_type, annotations in result['llm_annotations'].items():
            print(f"\n{annotation_type}:")
            if isinstance(annotations, dict):
                print(json.dumps(annotations, indent=2)[:800] + "...")


def example_model_comparison():
    """Example of comparing different Gemini models."""
    print("\n=== Model Comparison Example ===")
    
    config = setup_environment()
    if not config:
        return
    
    text = "We tentatively suggest that this approach might be somewhat effective, though further validation is clearly necessary."
    
    models_to_test = [
        GeminiModel.GEMINI_2_5_FLASH,
        GeminiModel.GEMINI_2_0_FLASH,
        # GeminiModel.GEMINI_2_5_PRO  # Uncomment if you have access
    ]
    
    results = {}
    
    for model in models_to_test:
        print(f"\nTesting with {model.value}...")
        
        # Update configuration for this model
        config.model_config.model = model
        annotator = AdvancedAnnotator(config)
        
        result = annotator.annotate_text(
            text=text,
            text_id=f"comparison_{model.value}",
            annotation_types=[AnnotationType.HEDGING_BOOSTING],
            include_nlp_analysis=False
        )
        
        results[model.value] = {
            "processing_time": result['metadata']['processing_time'],
            "features_detected": len(result.get('combined_features', [])),
            "model_used": result['metadata']['model_used']
        }
        
        print(f"  Processing time: {result['metadata']['processing_time']:.2f}s")
        print(f"  Features detected: {len(result.get('combined_features', []))}")
    
    print(f"\n--- Model Comparison Summary ---")
    print(json.dumps(results, indent=2))


def example_synthetic_data_generation():
    """Example of generating synthetic texts with specific linguistic features."""
    print("\n=== Synthetic Data Generation Example ===")
    
    config = setup_environment()
    if not config:
        return
    
    # Create synthetic data generator
    generator = SyntheticDataGenerator(config)
    
    print("Generating synthetic academic texts...")
    
    # Example 1: Generate a single text with specific parameters
    params = GenerationParameters(
        genre=TextGenreType.ACADEMIC_PAPER,
        complexity=LinguisticComplexity.MODERATE,
        target_features=["hedging", "academic_stance", "evaluative_language"],
        word_count_range=(150, 250),
        topic_area="computational linguistics",
        stance_type="tentative",
        author_persona="graduate student",
        include_citations=True
    )
    
    print("\nGenerating single text with specific features...")
    result = generator.generate_text(params)
    
    print(f"Generated text (ID: {result.generation_id}):")
    print(f"Word count: {result.metadata.get('word_count', 0)}")
    print(f"Quality score: {result.quality_metrics.get('overall_quality', 0):.2f}")
    print(f"Text preview: {result.text[:200]}...")
    
    if result.actual_features:
        print(f"\nDetected features:")
        for feature_type, count in result.actual_features.get('feature_types', {}).items():
            print(f"  {feature_type}: {count}")
    
    # Example 2: Generate a small dataset with variety
    print(f"\nGenerating dataset of academic texts...")
    
    dataset = generate_academic_texts(
        config=config,
        count=3,
        complexity=LinguisticComplexity.MODERATE
    )
    
    print(f"Generated {len(dataset)} texts in dataset:")
    for i, text_obj in enumerate(dataset, 1):
        print(f"  Text {i}: {len(text_obj.text.split())} words, "
              f"quality: {text_obj.quality_metrics.get('overall_quality', 0):.2f}")
    
    # Example 3: Generate business emails with politeness focus
    print(f"\nGenerating business emails with politeness variations...")
    
    email_params = [
        GenerationParameters(
            genre=TextGenreType.BUSINESS_EMAIL,
            complexity=LinguisticComplexity.SIMPLE,
            target_features=["negative_politeness", "hedging"],
            word_count_range=(80, 120),
            topic_area="project update request",
            politeness_level="formal"
        ),
        GenerationParameters(
            genre=TextGenreType.BUSINESS_EMAIL,
            complexity=LinguisticComplexity.SIMPLE,
            target_features=["positive_politeness", "directness"],
            word_count_range=(60, 100),
            topic_area="team meeting invitation",
            politeness_level="informal"
        )
    ]
    
    email_results = generator.generate_batch(email_params)
    
    for i, email in enumerate(email_results, 1):
        print(f"\nEmail {i} ({'formal' if i == 1 else 'informal'}):")
        print(f"  Features: {', '.join(email.parameters.target_features)}")
        print(f"  Preview: {email.text[:100]}...")
    
    # Show generator statistics
    stats = generator.get_statistics()
    print(f"\n--- Generation Statistics ---")
    print(f"Total generated: {stats['total_generated']}")
    print(f"Successful: {stats['successful_generations']}")
    print(f"Failed: {stats['failed_generations']}")


def example_annotation_plus_generation_workflow():
    """Example combining annotation analysis with synthetic data generation."""
    print("\n=== Combined Annotation + Generation Workflow ===")
    
    config = setup_environment()
    if not config:
        return
    
    # Step 1: Analyze existing text to understand its features
    annotator = AdvancedAnnotator(config)
    
    existing_text = """
    Our research demonstrates that hedging strategies in academic writing serve multiple 
    functions beyond mere politeness. We tentatively suggest that these linguistic choices 
    reflect deeper epistemological commitments. However, further investigation may be needed 
    to fully understand the implications. It is possible that previous studies have 
    underestimated the complexity of this phenomenon.
    """
    
    print("Step 1: Analyzing existing text...")
    analysis = annotator.annotate_text(
        text=existing_text,
        text_id="source_text",
        annotation_types=[AnnotationType.HEDGING_BOOSTING, AnnotationType.ACADEMIC_STANCE]
    )
    
    # Extract features from the analysis
    detected_features = []
    if 'llm_annotations' in analysis:
        for annotation_type, annotations in analysis['llm_annotations'].items():
            if isinstance(annotations, dict) and not annotations.get('error'):
                # Look for feature arrays in annotations
                for key, value in annotations.items():
                    if isinstance(value, list) and key.endswith('_features'):
                        detected_features.extend([f.get('category', 'unknown') for f in value if isinstance(f, dict)])
    
    print(f"Detected features in source text: {set(detected_features)}")
    
    # Step 2: Generate similar texts with the same features
    print("\nStep 2: Generating similar texts with detected features...")
    
    generator = SyntheticDataGenerator(config)
    
    # Create generation parameters based on the analysis
    generation_params = GenerationParameters(
        genre=TextGenreType.ACADEMIC_PAPER,
        complexity=LinguisticComplexity.MODERATE,
        target_features=list(set(detected_features))[:3],  # Use top 3 detected features
        word_count_range=(100, 200),
        topic_area="academic research methodology",
        stance_type="tentative",  # Based on hedging in source
        author_persona="researcher"
    )
    
    # Generate similar texts
    similar_texts = generator.generate_batch([generation_params] * 2)
    
    print(f"Generated {len(similar_texts)} similar texts:")
    for i, text_obj in enumerate(similar_texts, 1):
        print(f"\nGenerated Text {i}:")
        print(f"  Target features: {', '.join(text_obj.parameters.target_features)}")
        print(f"  Word count: {len(text_obj.text.split())}")
        print(f"  Preview: {text_obj.text[:150]}...")
    
    print("\n🔄 This workflow demonstrates how to:")
    print("1. Analyze existing texts to identify linguistic patterns")
    print("2. Use those patterns to generate new synthetic data")
    print("3. Create datasets with controlled linguistic features")
    print("4. Augment training data for machine learning models")


def main():
    """Run all examples."""
    print("Advanced Document Annotation Tool - Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_single_text_annotation()
        example_csv_annotation()
        example_custom_prompt_chain()
        example_model_comparison()
        example_synthetic_data_generation()
        example_annotation_plus_generation_workflow()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Please ensure you have:")
        print("1. Set your GOOGLE_API_KEY environment variable")
        print("2. Installed all required dependencies (pip install -r requirements.txt)")
        print("3. Downloaded the spaCy language model (python -m spacy download en_core_web_lg)")


if __name__ == "__main__":
    main()
