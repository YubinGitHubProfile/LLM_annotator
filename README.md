# Advanced Document Annotation Tool

A sophisticated document annotation system leveraging **LangChain**, **Google Gemini models**, and **spaCy** for comprehensive socio-pragmatic analysis of academic and professional texts. This tool goes far beyond simple keyword matching to capture nuanced linguistic phenomena such as stance-taking, rapport-building, politeness strategies, and complex discourse patterns.

## 🚀 Key Features

### Advanced AI Integration
- **LangChain Framework**: Robust prompt management and model orchestration
- **Multiple Gemini Models**: Support for Gemini 2.5-Flash (default), 2.0-Flash, and 2.5-Pro
- **Intelligent Prompt Chaining**: Multi-step analysis for comprehensive annotation

### Sophisticated Linguistic Analysis
- **Socio-Pragmatic Features**: Interpersonal positioning, face-work strategies, social distance markers
- **Academic Stance Analysis**: Epistemic stance, evaluative language, dialogic positioning
- **Rapport Building Detection**: Solidarity markers, engagement strategies, positive face work
- **Politeness Strategies**: Brown & Levinson's politeness theory implementation
- **Hedging & Boosting**: Commitment modulation and claim strength analysis
- **Discourse Organization**: Coherence markers, transition signals, structural patterns

### Enhanced NLP Capabilities
- **spaCy Integration**: Advanced grammatical and semantic feature extraction
- **Readability Metrics**: Comprehensive text complexity analysis
- **Citation Pattern Detection**: Academic reference analysis
- **Nominalization & Passive Voice**: Formal register identification
- **Named Entity Recognition**: Context-aware entity extraction

### Synthetic Data Generation
- **Genre-Specific Creation**: Academic papers, business emails, social media posts
- **Controllable Linguistic Features**: Complexity levels, politeness strategies, stance types
- **Quality Metrics**: Readability scores, feature validation, coherence analysis
- **Batch Generation**: Create datasets with specified characteristics
- **Combined Workflows**: Generate and annotate in unified pipelines

### Flexible Data Processing
- **Multiple Input Formats**: CSV, JSON, text files, and directory processing
- **Batch Processing**: Efficient handling of large document collections
- **Configurable Pipelines**: Customizable annotation workflows
- **Rich Output Formats**: JSON and CSV export with detailed annotations

## 🏗️ Architecture

The tool consists of several interconnected modules:

```
├── config/
│   └── settings.py              # Configuration management
├── agents/
│   └── advanced_annotator.py    # LangChain-based annotator
├── generators/
│   └── synthetic_data.py        # Synthetic text generation
├── prompts/
│   └── advanced_templates.py    # Sophisticated prompt templates
├── nlp/
│   └── spacy_analyzer.py       # spaCy-based linguistic analysis
├── data/
│   └── advanced_loader.py      # Enhanced data processing
├── examples/
│   └── usage_examples.py       # Comprehensive usage examples
└── requirements.txt            # Dependencies
```

## 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YubinGitHubProfile/LLM_annotator.git
cd LLM_annotator
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download spaCy language model:**
```bash
python -m spacy download en_core_web_lg
```

4. **Set up your Google API key:**
```bash
export GOOGLE_API_KEY="your_gemini_api_key_here"
```

## 🎯 Quick Start

### Basic Usage

```python
from config.settings import Config, AnnotationType
from agents.advanced_annotator import AdvancedAnnotator
from prompts.advanced_templates import COMPREHENSIVE_ANALYSIS_CHAIN

# Initialize configuration
config = Config(api_key="your_api_key")

# Create annotator with comprehensive analysis
annotator = AdvancedAnnotator(config, prompt_chain=COMPREHENSIVE_ANALYSIS_CHAIN)

# Annotate a text
text = """
We argue that this approach represents a significant advancement in the field. 
However, further research may be needed to validate these findings. 
In our view, the implications are substantial and warrant careful consideration.
"""

result = annotator.annotate_text(
    text=text,
    text_id="sample_1",
    include_nlp_analysis=True
)

print(f"Detected {len(result['combined_features'])} linguistic features")
```

### CSV File Processing

```python
from data.advanced_loader import AdvancedDataLoader

# Create data loader
data_loader = AdvancedDataLoader(config, annotator)

# Process CSV file
summary = data_loader.annotate_csv(
    input_file="academic_texts.csv",
    output_file="annotated_results.json",
    text_column="text",
    id_column="document_id",
    annotation_types=[
        AnnotationType.ACADEMIC_STANCE,
        AnnotationType.HEDGING_BOOSTING
    ]
)
```

### Synthetic Data Generation

```python
from generators.synthetic_data import SyntheticDataGenerator, TextGenreType, LinguisticComplexity

# Initialize generator
generator = SyntheticDataGenerator(config)

# Generate academic text
academic_text = generator.generate_text(
    genre=TextGenreType.ACADEMIC,
    topic="climate change research methodology",
    complexity=LinguisticComplexity.INTERMEDIATE,
    target_features=['hedging', 'stance_markers', 'citations']
)

# Generate business email
business_email = generator.generate_text(
    genre=TextGenreType.BUSINESS_EMAIL,
    topic="project proposal request",
    complexity=LinguisticComplexity.BEGINNER,
    target_features=['politeness', 'formal_register']
)

# Batch generation for dataset creation
dataset = generator.generate_dataset(
    count=50,
    genre=TextGenreType.ACADEMIC,
    topics=["machine learning", "linguistics", "psychology"],
    complexity_range=LinguisticComplexity.INTERMEDIATE
)
```

### Combined Annotation and Generation Workflow

```python
# Generate synthetic text and immediately annotate it
synthetic_text = generator.generate_text(
    genre=TextGenreType.ACADEMIC,
    topic="natural language processing",
    complexity=LinguisticComplexity.ADVANCED
)

# Annotate the generated text
annotation_result = annotator.annotate_text(
    text=synthetic_text.content,
    text_id="synthetic_001"
)

# Validate generation quality
quality_metrics = generator.evaluate_quality(synthetic_text.content)
print(f"Generated text quality score: {quality_metrics['overall_score']}")
```

## 🔧 Advanced Configuration

### Model Selection

```python
from config.settings import GeminiModel

# High-performance model (default)
config.model_config.model = GeminiModel.GEMINI_2_5_FLASH

# Experimental model
config.model_config.model = GeminiModel.GEMINI_2_0_FLASH

# Most capable model
config.model_config.model = GeminiModel.GEMINI_2_5_PRO
```

### Custom Annotation Types

```python
from config.settings import AnnotationType

# Select specific annotation types
config.annotation_config.annotation_types = [
    AnnotationType.SOCIO_PRAGMATIC,
    AnnotationType.POLITENESS_STRATEGIES,
    AnnotationType.RAPPORT_BUILDING
]
```

### Custom Prompt Chains

```python
from prompts.advanced_templates import PromptChain, PromptType

# Create focused analysis chain
academic_chain = PromptChain([
    PromptType.ACADEMIC_STANCE,
    PromptType.HEDGING_BOOSTING
])

annotator = AdvancedAnnotator(config, prompt_chain=academic_chain)
```

## 📊 Annotation Types

### 1. Socio-Pragmatic Analysis
- **Interpersonal Positioning**: Power relations, authority claims, social distance
- **Face Work Strategies**: Face-threatening acts, mitigation strategies
- **Identity Construction**: In-group/out-group positioning, role establishment

### 2. Academic Stance Analysis
- **Epistemic Stance**: Certainty markers, evidence presentation, knowledge claims
- **Evaluative Stance**: Value judgments, significance markers, novelty claims
- **Dialogic Stance**: Citation integration, agreement/disagreement patterns

### 3. Rapport Building
- **Solidarity Markers**: Inclusive pronouns, shared knowledge assumptions
- **Engagement Strategies**: Direct address, rhetorical questions, interactive elements
- **Positive Face Work**: Compliments, recognition, validation

### 4. Politeness Strategies
- **Positive Politeness**: In-group markers, agreement seeking, humor
- **Negative Politeness**: Hedging, deference, minimizing imposition
- **Off-Record**: Indirectness, metaphors, ambiguity

### 5. Hedging & Boosting
- **Hedging**: Modal verbs, epistemic adverbs, conditional constructions
- **Boosting**: Strong modals, intensifiers, categorical assertions

## 🎯 Use Cases

### Academic Writing Analysis
- **Thesis/Dissertation Review**: Identify stance patterns and argumentation strategies
- **Journal Article Analysis**: Examine author positioning and scholarly discourse
- **Student Writing Assessment**: Evaluate academic register and persuasive techniques

### Professional Communication
- **Business Writing**: Analyze rapport-building and politeness in corporate communication
- **Grant Proposals**: Examine persuasive strategies and confidence markers
- **Policy Documents**: Identify hedging patterns and authority construction

### Discourse Research
- **Corpus Linguistics**: Large-scale analysis of linguistic phenomena
- **Sociolinguistic Studies**: Examine language variation and social positioning
- **Pragmatic Analysis**: Study context-dependent meaning construction

## 📈 Output Format

The tool provides rich, structured output:

```json
{
  "text_id": "sample_1",
  "original_text": "...",
  "llm_annotations": {
    "socio_pragmatic": {
      "interpersonal_style": "formal_academic",
      "authority_level": "high",
      "features": [...]
    },
    "academic_stance": {
      "epistemic_stance": "confident",
      "evaluative_stance": "positive",
      "features": [...]
    }
  },
  "nlp_analysis": {
    "features": [...],
    "readability_scores": {...},
    "discourse_structure": {...}
  },
  "combined_features": [...],
  "metadata": {
    "processing_time": 2.34,
    "model_used": "gemini-2.5-flash",
    "confidence_scores": {...}
  }
}
```

## 🔬 Research Applications

This tool is designed for researchers studying:
- **Computational Sociolinguistics**: Automated analysis of social positioning in text
- **Digital Humanities**: Large-scale analysis of historical and literary texts
- **Applied Linguistics**: Study of language use in specific contexts and genres
- **Writing Studies**: Analysis of rhetorical strategies and discourse patterns
- **Communication Research**: Examination of persuasive and interpersonal strategies

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines for:
- Adding new annotation types
- Expanding prompt templates
- Improving NLP analysis
- Adding support for new languages/models

## 📝 Citation

If you use this tool in your research, please cite:

```bibtex
@software{advanced_annotation_tool,
  title={Advanced Document Annotation Tool for Socio-Pragmatic Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/YubinGitHubProfile/LLM_annotator}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
