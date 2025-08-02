# Project Structure

## Advanced Document Annotation Tool v2.0

This is a complete sophisticated document annotation system for socio-pragmatic analysis.

### Core Components

```
├── setup.py                    # Installation and setup script
├── start.py                    # Quick start initialization
├── requirements.txt            # Python dependencies
├── __init__.py                # Main package initialization
│
├── config/                     # Configuration management
│   ├── __init__.py
│   └── settings.py            # Comprehensive configuration classes
│
├── agents/                     # Advanced annotation agents
│   ├── __init__.py
│   └── advanced_annotator.py  # LangChain-based sophisticated annotator
│
├── prompts/                    # Advanced prompt templates
│   ├── __init__.py
│   └── advanced_templates.py  # Expert-level prompt engineering
│
├── nlp/                       # spaCy-based linguistic analysis
│   ├── __init__.py
│   └── spacy_analyzer.py      # Advanced linguistic feature extraction
│
├── data/                      # Enhanced data processing
│   ├── __init__.py
│   └── advanced_loader.py     # Multi-format data loading & processing
│
├── examples/                  # Comprehensive usage examples
│   ├── __init__.py
│   └── usage_examples.py      # Detailed demonstrations
│
└── sentiment_dataset/         # Sample data for testing
    └── sentiment_analysis.csv
```

### Key Features

- **LangChain Integration**: Professional prompt management and model orchestration
- **Multiple Gemini Models**: 2.5-Flash (default), 2.0-Flash, 2.5-Pro support
- **Advanced spaCy Analysis**: Sophisticated linguistic feature detection
- **Socio-Pragmatic Focus**: Beyond simple annotation to nuanced analysis
- **Academic Discourse**: Specialized tools for scholarly text analysis
- **Flexible Processing**: Multiple input formats and batch processing

### Quick Start

1. Install: `python setup.py`
2. Configure: Set your `GOOGLE_API_KEY`
3. Run: `python examples/usage_examples.py`

### Annotation Capabilities

- **Interpersonal Positioning**: Authority, social distance, power relations
- **Academic Stance**: Epistemic, evaluative, dialogic positioning
- **Politeness Strategies**: Brown & Levinson framework implementation  
- **Rapport Building**: Solidarity markers and engagement strategies
- **Hedging & Boosting**: Commitment modulation and claim strength
- **Discourse Organization**: Coherence markers and structural patterns

This system represents a significant advancement from basic keyword-based annotation to sophisticated socio-pragmatic analysis suitable for academic research and professional applications.
