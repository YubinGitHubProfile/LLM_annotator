"""
Setup script for the Advanced Document Annotation Tool.
Helps users install dependencies and configure the environment.
"""

import subprocess
import sys
import os
from pathlib import Path
import pkg_resources
from typing import List, Tuple


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= min_version:
        print(f"✓ Python {'.'.join(map(str, current_version))} is compatible")
        return True
    else:
        print(f"✗ Python {'.'.join(map(str, min_version))} or higher is required")
        print(f"  Current version: {'.'.join(map(str, current_version))}")
        return False


def install_requirements() -> bool:
    """Install required packages from requirements.txt."""
    print("\n📦 Installing Python packages...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✓ Python packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install packages: {e}")
        return False


def install_spacy_model() -> bool:
    """Install spaCy language model."""
    print("\n🔤 Installing spaCy language model...")
    
    model_name = "en_core_web_lg"
    
    try:
        # Check if model is already installed
        import spacy
        try:
            nlp = spacy.load(model_name)
            print(f"✓ spaCy model '{model_name}' is already installed")
            return True
        except OSError:
            pass
    except ImportError:
        print("  spaCy not installed yet, will install with model...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", model_name
        ])
        print(f"✓ spaCy model '{model_name}' installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install spaCy model: {e}")
        print(f"  Try running manually: python -m spacy download {model_name}")
        return False


def check_installed_packages() -> List[Tuple[str, bool]]:
    """Check which required packages are installed."""
    required_packages = [
        "langchain",
        "langchain-google-genai", 
        "google-generativeai",
        "spacy",
        "pandas",
        "numpy",
        "python-dotenv",
        "pydantic",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "textstat",
        "nltk",
        "transformers",
        "sentence-transformers"
    ]
    
    results = []
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
            results.append((package, True))
        except pkg_resources.DistributionNotFound:
            results.append((package, False))
    
    return results


def setup_environment_file():
    """Create a template .env file for environment variables."""
    print("\n🔧 Setting up environment configuration...")
    
    env_file = Path(__file__).parent / ".env"
    
    if env_file.exists():
        print("✓ .env file already exists")
        return True
    
    env_template = """# Google API Key for Gemini models
# Get your API key from: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your_api_key_here

# Optional: Custom configurations
# GEMINI_MODEL=gemini-2.5-flash
# BATCH_SIZE=5
# TEMPERATURE=0.1
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_template)
        print("✓ Created .env template file")
        print("  Please edit .env and add your Google API key")
        return True
    except Exception as e:
        print(f"✗ Failed to create .env file: {e}")
        return False


def create_sample_config():
    """Create a sample configuration file."""
    print("\n⚙️ Creating sample configuration...")
    
    config_dir = Path(__file__).parent / "config"
    config_dir.mkdir(exist_ok=True)
    
    sample_config_file = config_dir / "sample_config.json"
    
    sample_config = {
        "model_config": {
            "model": "gemini-2.5-flash",
            "temperature": 0.1,
            "max_output_tokens": 8000,
            "top_p": 0.95,
            "top_k": 40,
            "candidate_count": 1
        },
        "spacy_config": {
            "model_name": "en_core_web_lg",
            "components": [
                "tok2vec", "tagger", "parser", "attribute_ruler", 
                "lemmatizer", "ner"
            ],
            "custom_components": [
                "sentiment", "textcat", "entity_ruler"
            ]
        },
        "annotation_config": {
            "batch_size": 5,
            "max_retries": 3,
            "retry_delay": 1.0,
            "include_confidence_scores": True,
            "enable_preprocessing": True,
            "enable_postprocessing": True,
            "annotation_types": [
                "socio_pragmatic",
                "academic_stance", 
                "rapport_building",
                "hedging_boosting",
                "politeness_strategies"
            ]
        }
    }
    
    try:
        import json
        with open(sample_config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        print("✓ Created sample configuration file")
        return True
    except Exception as e:
        print(f"✗ Failed to create sample config: {e}")
        return False


def test_installation():
    """Test if the installation is working correctly."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test basic imports
        print("  Testing imports...")
        
        # Test spaCy
        import spacy
        nlp = spacy.load("en_core_web_lg")
        print("    ✓ spaCy working")
        
        # Test Google API (basic import)
        import google.generativeai as genai
        print("    ✓ Google AI SDK working")
        
        # Test LangChain
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("    ✓ LangChain Google AI working")
        
        # Test other key packages
        import pandas as pd
        import numpy as np
        import textstat
        print("    ✓ Other packages working")
        
        print("✓ All core components are working!")
        return True
        
    except ImportError as e:
        print(f"    ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"    ✗ Unexpected error: {e}")
        return False


def display_next_steps():
    """Display next steps for the user."""
    print("\n🎯 Next Steps:")
    print("1. Edit the .env file and add your Google API key")
    print("   Get your key from: https://aistudio.google.com/app/apikey")
    print()
    print("2. Run the example script to test the tool:")
    print("   python examples/usage_examples.py")
    print()
    print("3. Check out the documentation in README.md")
    print()
    print("4. Start annotating your documents!")
    print()
    print("🔗 Useful Resources:")
    print("   - Google AI Studio: https://aistudio.google.com/")
    print("   - spaCy Documentation: https://spacy.io/")
    print("   - LangChain Documentation: https://python.langchain.com/")


def main():
    """Main setup function."""
    print("🚀 Advanced Document Annotation Tool - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\nPlease upgrade Python and run setup again.")
        return False
    
    # Show current package status
    print("\n📋 Checking current package installation...")
    packages = check_installed_packages()
    installed = sum(1 for _, is_installed in packages if is_installed)
    total = len(packages)
    
    print(f"Installed packages: {installed}/{total}")
    for package, is_installed in packages:
        status = "✓" if is_installed else "✗"
        print(f"  {status} {package}")
    
    # Install requirements
    if installed < total:
        if not install_requirements():
            print("\nSetup failed. Please check the error messages above.")
            return False
    else:
        print("\n✓ All packages are already installed")
    
    # Install spaCy model
    if not install_spacy_model():
        print("\nWarning: spaCy model installation failed.")
        print("The tool may not work properly without the language model.")
    
    # Setup environment file
    setup_environment_file()
    
    # Create sample configuration
    create_sample_config()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        display_next_steps()
        return True
    else:
        print("\n❌ Setup completed with errors.")
        print("Please check the error messages and resolve any issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
