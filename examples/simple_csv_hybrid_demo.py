#!/usr/bin/env python3
"""
Simple CSV-based hybrid rapport analysis demo.
Processes 20 texts with LLM + spaCy keyword matching, outputs to CSV.
"""

import sys
import os
import pandas as pd
import csv
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Config
from agents.advanced_annotator import AdvancedAnnotator
from nlp.spacy_analyzer import AdvancedNLPAnalyzer
from prompts.advanced_templates import PromptType


def create_sample_data():
    """Create a sample CSV file with 20 texts for testing."""
    
    # Ensure production directory exists
    production_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'production')
    os.makedirs(production_dir, exist_ok=True)
    
    sample_texts = [
        "We hope this research will help us understand the challenges together.",
        "Thank you for your valuable insights during our last meeting.",
        "Perhaps we could schedule a brief call to explore these ideas.",
        "I believe this work will advance our field significantly.",
        "Our findings suggest that collaboration is essential for breakthroughs.",
        "Follow these steps carefully to ensure proper installation.",
        "The configuration process requires administrative privileges.",
        "Contact support if problems persist after troubleshooting.",
        "This section describes the technical specifications in detail.",
        "Users must comply with all security protocols and guidelines.",
        "We appreciate your continued partnership in this endeavor.",
        "I hope you find these recommendations useful for your project.",
        "Together, we can overcome the obstacles facing our industry.",
        "Your feedback has been invaluable to our development process.",
        "Let's work together to identify the best solution for everyone.",
        "The system will automatically generate reports as scheduled.",
        "Error messages will appear in the console window for debugging.",
        "Database connections must be properly configured before use.",
        "We're excited to share our latest research findings with you.",
        "I trust that our collaboration will yield positive results."
    ]
    
    # Create DataFrame
    data = {
        'ID': [f'text_{i+1:02d}' for i in range(20)],
        'text': sample_texts
    }
    
    df = pd.DataFrame(data)
    input_file = os.path.join(production_dir, 'sample_rapport_texts.csv')
    df.to_csv(input_file, index=False)
    print(f"✅ Created sample CSV file: {input_file}")
    return input_file


def process_csv_hybrid_analysis(input_file: str, output_file: str):
    """Process CSV file with hybrid LLM + spaCy analysis."""
    
    print("🤝 Simple CSV Hybrid Rapport Analysis")
    print("=" * 50)
    
    # Initialize components
    config = Config()
    # Set Gemini 2.5 Flash with temperature 0.1
    from config.settings import GeminiModel
    config.model_config.model = GeminiModel.GEMINI_2_5_FLASH
    config.model_config.temperature = 0.1
    
    annotator = AdvancedAnnotator(config)
    spacy_analyzer = AdvancedNLPAnalyzer(config.spacy_config)
    
    # Define rapport keywords
    rapport_keywords = [
        "we", "us", "our", "together",
        ("thank", "you"), ("I", "hope"), ("I", "believe"),
        ("perhaps", "we"), ("work", "together"),
        (("valuable", "ADJ"), ("insights", "NOUN")),
        "appreciate", "collaboration", "partnership"
    ]
    
    # Read input CSV
    print(f"📖 Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Found {len(df)} texts to process")
    
    # Prepare output data
    output_data = []
    batch_size = 10
    
    # Token tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    
    # Process in batches
    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        print(f"\n🔄 Processing batch {batch_start//batch_size + 1}: rows {batch_start+1}-{batch_end}")
        
        for _, row in batch_df.iterrows():
            text_id = row['ID']
            text = row['text']
            
            # print(f"  📝 Processing {text_id}...")
            
            # 1. LLM Analysis
            llm_label = 0
            try:
                llm_result = annotator.annotate_text(
                    text,
                    annotation_types=[PromptType.RAPPORT_BUILDING],
                    include_nlp_analysis=False
                )
                
                # Track token usage
                if hasattr(llm_result, 'token_count'):
                    if isinstance(llm_result.token_count, dict):
                        input_tokens = llm_result.token_count.get('input_tokens', 0)
                        output_tokens = llm_result.token_count.get('output_tokens', 0)
                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens
                        total_tokens += input_tokens + output_tokens
                    elif isinstance(llm_result.token_count, (int, float)):
                        total_tokens += llm_result.token_count
                elif hasattr(llm_result, 'usage'):
                    # Alternative token tracking structure
                    usage = llm_result.usage
                    if hasattr(usage, 'input_tokens'):
                        total_input_tokens += usage.input_tokens
                    if hasattr(usage, 'output_tokens'):
                        total_output_tokens += usage.output_tokens
                    if hasattr(usage, 'total_tokens'):
                        total_tokens += usage.total_tokens
                elif isinstance(llm_result, dict) and 'usage' in llm_result:
                    # Dict structure with usage
                    usage = llm_result['usage']
                    total_input_tokens += usage.get('input_tokens', 0)
                    total_output_tokens += usage.get('output_tokens', 0)
                    total_tokens += usage.get('total_tokens', 0)
                else:
                    # Estimate tokens based on text length (rough approximation: ~4 chars per token)
                    estimated_input = len(text) // 4
                    estimated_output = 50  # Rough estimate for typical response
                    total_input_tokens += estimated_input
                    total_output_tokens += estimated_output
                    total_tokens += estimated_input + estimated_output
                
                # Parse LLM response for rapport indicators
                if hasattr(llm_result, 'annotations') and 'rapport_building' in llm_result.annotations:
                    llm_label = 1
                elif isinstance(llm_result, dict):
                    if 'rapport_building' in llm_result:
                        llm_label = 1
                    elif 'text' in llm_result:
                        response_text = llm_result['text'].lower()
                        rapport_indicators = ['rapport', 'relationship', 'connection', 'collaborative', 'we', 'us', 'together']
                        if any(indicator in response_text for indicator in rapport_indicators):
                            llm_label = 1
                elif isinstance(llm_result, str):
                    response_text = llm_result.lower()
                    rapport_indicators = ['rapport', 'relationship', 'connection', 'collaborative', 'we', 'us', 'together']
                    if any(indicator in response_text for indicator in rapport_indicators):
                        llm_label = 1
                        
            except Exception as e:
                print(f"    ❌ LLM failed for {text_id}: {e}")
                llm_label = 0
            
            # 2. spaCy Keyword Analysis
            keyword_result = spacy_analyzer.classify_by_keywords(
                text,
                rapport_keywords,
                feature_name="rapport_building",
                fuzzy_threshold=0.8,
                enable_fuzzy=True
            )
            
            spacy_label = keyword_result['label']
            
            # 3. Extract matched keywords with POS tags
            keywords_matched = []
            if keyword_result['match_details']:
                for detail in keyword_result['match_details']:
                    if detail['type'] == 'simple':
                        keyword = detail['keyword']
                        keywords_matched.append(keyword)
                    else:
                        # For tuples, show the structure with POS tags
                        keyword_tuple = detail['keyword']
                        if any('required_pos' in match for match in detail['matches']):
                            # POS-constrained tuple
                            pos_info = []
                            for match in detail['matches']:
                                if 'required_pos' in match:
                                    pos_info.append(f"{match['text']}({match['pos']})")
                                else:
                                    pos_info.append(match['text'])
                            keywords_matched.append(f"({', '.join(pos_info)})")
                        else:
                            # Simple tuple
                            tuple_texts = [match['text'] for match in detail['matches']]
                            keywords_matched.append(f"({', '.join(tuple_texts)})")
            
            # Join keywords or use "none" if empty
            keywords_str = "; ".join(keywords_matched) if keywords_matched else "none"
            
            # Store result
            output_data.append({
                'ID': text_id,
                'text': text,
                'llm_label': llm_label,
                'spacy_label': spacy_label,
                'keywords_matched': keywords_str
            })
            
            # print(f"    ✅ {text_id}: LLM={llm_label}, spaCy={spacy_label}, Keywords={len(keywords_matched)}")
    
    # Save results to CSV
    print(f"\n💾 Saving results to: {output_file}")
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)
    
    # Calculate summary statistics
    total_texts = len(output_df)
    llm_positive = sum(output_df['llm_label'])
    spacy_positive = sum(output_df['spacy_label'])
    agreement_rate = sum(output_df['llm_label'] == output_df['spacy_label']) / total_texts * 100
    
    # Print summary
    print(f"\n📊 Analysis Summary:")
    print(f"Total texts processed: {total_texts}")
    print(f"LLM positive labels: {llm_positive}")
    print(f"spaCy positive labels: {spacy_positive}")
    print(f"Agreement rate: {agreement_rate:.1f}%")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total tokens used: {total_tokens}")
    
    # Save statistics to CSV
    stats_data = {
        'Metric': [
            'Total texts processed',
            'LLM positive labels',
            'spaCy positive labels', 
            'Agreement rate (%)',
            'LLM precision (%)',
            'spaCy precision (%)',
            'Both positive',
            'Both negative',
            'Total input tokens',
            'Total output tokens', 
            'Total tokens used',
            'Avg tokens per text',
            'LLM model',
            'Temperature',
            'Max output tokens',
            'Top P',
            'Top K'
        ],
        'Value': [
            total_texts,
            llm_positive,
            spacy_positive,
            round(agreement_rate, 1),
            round((llm_positive / total_texts * 100) if total_texts > 0 else 0, 1),
            round((spacy_positive / total_texts * 100) if total_texts > 0 else 0, 1),
            sum((output_df['llm_label'] == 1) & (output_df['spacy_label'] == 1)),
            sum((output_df['llm_label'] == 0) & (output_df['spacy_label'] == 0)),
            total_input_tokens,
            total_output_tokens,
            total_tokens,
            round((total_tokens / total_texts) if total_texts > 0 else 0, 1),
            config.model_config.model.value,
            config.model_config.temperature,
            config.model_config.max_output_tokens,
            config.model_config.top_p,
            config.model_config.top_k
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    production_dir = os.path.dirname(output_file)  # Get the production directory from output_file path
    stats_file = os.path.join(production_dir, 'annotation_stats.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"📊 Statistics saved to: {stats_file}")
    
    return output_df


def main():
    """Main function to run the simple CSV hybrid analysis."""
    
    # Get production directory path
    production_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'production')
    
    # Create sample data
    input_file = create_sample_data()
    
    # Process with hybrid analysis - save outputs to production folder
    output_file = os.path.join(production_dir, 'hybrid_rapport_results.csv')
    results = process_csv_hybrid_analysis(input_file, output_file)
    
    # Show first few results
    print(f"\n📋 Sample Results:")
    print(results.head(5).to_string(index=False))
    
    print(f"\n✅ Analysis complete! Check production folder for all results:")
    print(f"  - Input: {input_file}")
    print(f"  - Output: {output_file}")
    print(f"  - Stats: {os.path.join(production_dir, 'annotation_stats.csv')}")


if __name__ == "__main__":
    main()
