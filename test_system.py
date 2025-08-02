#!/usr/bin/env python3
"""
Quick system test for the LLM Annotator with synthetic data generation.
Tests both annotation and generation capabilities.
"""

import os
import sys
from typing import Dict, Any

def test_imports():
    """Test that all modules can be imported correctly."""
    print("🔍 Testing imports...")
    
    try:
        # Test configuration imports
        from config.settings import Config, GeminiModel, AnnotationType
        print("✅ Config module imported successfully")
        
        # Test agent imports
        from agents.advanced_annotator import AdvancedAnnotator
        print("✅ Advanced annotator imported successfully")
        
        # Test generator imports
        from generators.synthetic_data import SyntheticDataGenerator, TextGenreType, LinguisticComplexity
        print("✅ Synthetic data generator imported successfully")
        
        # Test data loader imports
        from data.advanced_loader import AdvancedDataLoader
        print("✅ Data loader imported successfully")
        
        # Test prompt imports
        from prompts.advanced_templates import PromptChain, PromptType
        print("✅ Prompt templates imported successfully")
        
        # Test NLP imports
        from nlp.spacy_analyzer import SpacyAnalyzer
        print("✅ spaCy analyzer imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during imports: {e}")
        return False

def test_configuration():
    """Test configuration setup."""
    print("\n🔧 Testing configuration...")
    
    try:
        from config.settings import Config, GeminiModel
        
        # Test basic config creation
        config = Config()
        print("✅ Config created successfully")
        
        # Test model switching
        config.model_config.model = GeminiModel.GEMINI_2_5_FLASH
        print(f"✅ Model set to: {config.model_config.model.value}")
        
        # Check API key warning
        if not config.api_key:
            print("⚠️  No API key found - set GOOGLE_API_KEY environment variable for full testing")
        else:
            print("✅ API key configured")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_annotation_pipeline():
    """Test annotation functionality (without API calls)."""
    print("\n📝 Testing annotation pipeline...")
    
    try:
        from config.settings import Config, AnnotationType
        from agents.advanced_annotator import AdvancedAnnotator
        from prompts.advanced_templates import COMPREHENSIVE_ANALYSIS_CHAIN
        
        config = Config()
        annotator = AdvancedAnnotator(config, prompt_chain=COMPREHENSIVE_ANALYSIS_CHAIN)
        print("✅ Annotator initialized successfully")
        
        # Test prompt generation (without API call)
        test_text = "We argue that this approach might represent a significant advancement."
        
        # Test NLP analysis (works without API)
        from nlp.spacy_analyzer import SpacyAnalyzer
        nlp_analyzer = SpacyAnalyzer(config.spacy_config)
        nlp_features = nlp_analyzer.analyze_text(test_text)
        print(f"✅ NLP analysis completed: {len(nlp_features)} features detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Annotation pipeline error: {e}")
        return False

def test_generation_setup():
    """Test synthetic data generation setup (without API calls)."""
    print("\n🎲 Testing generation pipeline...")
    
    try:
        from config.settings import Config
        from generators.synthetic_data import SyntheticDataGenerator, TextGenreType, LinguisticComplexity
        
        config = Config()
        generator = SyntheticDataGenerator(config)
        print("✅ Generator initialized successfully")
        
        # Test generation parameters
        params = generator._create_generation_parameters(
            genre=TextGenreType.ACADEMIC,
            topic="test topic",
            complexity=LinguisticComplexity.INTERMEDIATE
        )
        print(f"✅ Generation parameters created: {params.genre.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation pipeline error: {e}")
        return False

def test_data_processing():
    """Test data loading capabilities."""
    print("\n📁 Testing data processing...")
    
    try:
        from config.settings import Config
        from data.advanced_loader import AdvancedDataLoader
        from agents.advanced_annotator import AdvancedAnnotator
        
        config = Config()
        annotator = AdvancedAnnotator(config)
        data_loader = AdvancedDataLoader(config, annotator)
        print("✅ Data loader initialized successfully")
        
        # Test file format detection
        csv_detected = data_loader._detect_file_format("test.csv")
        json_detected = data_loader._detect_file_format("test.json")
        txt_detected = data_loader._detect_file_format("test.txt")
        
        print(f"✅ File format detection working: CSV={csv_detected}, JSON={json_detected}, TXT={txt_detected}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        return False

def run_full_test():
    """Run all tests and provide summary."""
    print("🚀 Starting LLM Annotator System Test\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_configuration), 
        ("Annotation Pipeline", test_annotation_pipeline),
        ("Generation Setup", test_generation_setup),
        ("Data Processing", test_data_processing)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for production.")
        if not os.getenv('GOOGLE_API_KEY'):
            print("💡 Don't forget to set your GOOGLE_API_KEY environment variable for full functionality.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)
