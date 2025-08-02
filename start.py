#!/usr/bin/env python3
"""
Quick initialization script for the Advanced Document Annotation Tool.
Run this to get started quickly with the enhanced system.
"""

import sys
import os
from pathlib import Path

def main():
    """Quick setup and demonstration."""
    print("🚀 Advanced Document Annotation Tool")
    print("Quick Start Initialization")
    print("=" * 40)
    
    # Check if setup is needed
    print("\n📦 Checking installation...")
    try:
        import langchain
        import spacy
        import google.generativeai
        print("✓ Core dependencies appear to be installed")
        
        # Check spaCy model
        try:
            nlp = spacy.load("en_core_web_lg")
            print("✓ spaCy language model is available")
        except:
            print("⚠️ spaCy language model not found")
            print("  Run: python -m spacy download en_core_web_lg")
        
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("\nPlease run setup first:")
        print("  python setup.py")
        return
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("\n🔑 API Key Setup Needed")
        print("Please set your Google API key:")
        print("1. Get key from: https://aistudio.google.com/app/apikey")
        print("2. Set environment variable: export GOOGLE_API_KEY='your_key'")
        print("3. Or edit the .env file in this directory")
    else:
        print("✓ Google API key is configured")
    
    print("\n🎯 Ready to start!")
    print("\nTry these commands:")
    print("• Run examples: python examples/usage_examples.py")
    print("• Full setup: python setup.py")
    print("• Read docs: open README.md")
    
    print("\n📚 Key features of the enhanced system:")
    print("• LangChain integration for robust prompt management")
    print("• Multiple Gemini models (2.5-Flash, 2.0-Flash, 2.5-Pro)")
    print("• Advanced spaCy NLP analysis")
    print("• Socio-pragmatic feature detection")
    print("• Academic discourse analysis")
    print("• Sophisticated politeness and stance detection")
    
    # Quick demo if possible
    if api_key and api_key != "your_api_key_here":
        response = input("\nWould you like to run a quick demo? (y/N): ")
        if response.lower().startswith('y'):
            try:
                run_quick_demo()
            except Exception as e:
                print(f"Demo failed: {e}")
                print("Please run: python examples/usage_examples.py")


def run_quick_demo():
    """Run a minimal demo of the system."""
    print("\n🎮 Running Quick Demo...")
    
    try:
        from config.settings import Config, AnnotationType
        from agents.advanced_annotator import AdvancedAnnotator
        
        # Create minimal configuration
        config = Config()
        config.annotation_config.annotation_types = [AnnotationType.HEDGING_BOOSTING]
        
        # Create annotator
        annotator = AdvancedAnnotator(config)
        
        # Test text
        test_text = "We believe this approach might be somewhat effective, though further research is clearly needed."
        
        print(f"Analyzing: '{test_text}'")
        
        result = annotator.annotate_text(
            text=test_text,
            text_id="demo",
            include_nlp_analysis=True
        )
        
        print(f"\n✓ Analysis completed in {result['metadata']['processing_time']:.2f} seconds")
        print(f"✓ Detected {len(result.get('combined_features', []))} linguistic features")
        
        # Show some features
        features = result.get('combined_features', [])[:3]  # First 3 features
        if features:
            print("\nSample detected features:")
            for feature in features:
                print(f"  • {feature.get('feature_type', 'unknown')}: '{feature.get('text', '')}'")
        
        print("\n🎉 Demo successful! The system is working correctly.")
        print("Run 'python examples/usage_examples.py' for comprehensive examples.")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please run setup first: python setup.py")
    except Exception as e:
        print(f"Demo error: {e}")
        print("Check your API key and run: python examples/usage_examples.py")


if __name__ == "__main__":
    main()
