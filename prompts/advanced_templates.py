"""
Advanced prompt templates for sophisticated linguistic annotation using LangChain.
Specialized prompts for socio-pragmatic analysis and academic discourse features.
"""

from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage
from pydantic import BaseModel, Field
from enum import Enum
import json


class PromptType(Enum):
    """Types of annotation prompts."""
    SOCIO_PRAGMATIC = "socio_pragmatic"
    ACADEMIC_STANCE = "academic_stance"
    RAPPORT_BUILDING = "rapport_building"
    POLITENESS_STRATEGIES = "politeness_strategies"
    HEDGING_BOOSTING = "hedging_boosting"
    DISCOURSE_MARKERS = "discourse_markers"
    EVALUATIVE_LANGUAGE = "evaluative_language"
    CITATION_ANALYSIS = "citation_analysis"


class AnnotationSchema(BaseModel):
    """Schema for annotation output validation."""
    text_id: str = Field(description="Unique identifier for the text")
    original_text: str = Field(description="The original text being annotated")
    annotations: Dict[str, Any] = Field(description="Annotation results")
    confidence_scores: Optional[Dict[str, float]] = Field(description="Confidence scores for annotations")
    metadata: Optional[Dict[str, Any]] = Field(description="Additional metadata")


class AdvancedPromptTemplates:
    """Collection of advanced prompt templates for linguistic annotation."""
    
    # System prompt for setting up the annotation context
    ANNOTATION_SYSTEM_PROMPT = """You are an expert computational linguist and discourse analyst specializing in socio-pragmatic features of academic and professional texts. Your task is to perform sophisticated linguistic annotation that captures nuanced aspects of language use that go beyond simple keyword matching or surface-level analysis.

You have expertise in:
- Socio-pragmatic analysis and interpersonal meaning
- Academic discourse and stance-taking strategies
- Politeness theory and face-work strategies
- Hedging, boosting, and epistemic modality
- Rapport-building and relationship management in text
- Discourse organization and coherence markers
- Evaluative language and attitudinal positioning

Your annotations should be:
1. Theoretically grounded in linguistic research
2. Contextually sensitive to genre and register
3. Nuanced in recognizing implicit and explicit features
4. Consistent with established annotation frameworks
5. Accompanied by confidence scores and explanations

Always provide structured JSON output with detailed reasoning for your annotations."""

    # Socio-pragmatic analysis prompt
    SOCIO_PRAGMATIC_PROMPT = """Analyze the following text for socio-pragmatic features that reveal how the author manages interpersonal relationships, establishes authority, and positions themselves relative to their audience and content.

Text to analyze:
{text}

Focus on identifying and annotating:

1. **Interpersonal Positioning**:
   - Power relations and authority claims
   - Social distance markers (formal/informal language)
   - In-group/out-group positioning
   - Identity construction strategies

2. **Face Work Strategies**:
   - Face-threatening acts and their mitigation
   - Face-saving strategies
   - Positive and negative politeness markers
   - Directness vs. indirectness patterns

3. **Stance and Alignment**:
   - Epistemic stance (certainty, doubt, knowledge claims)
   - Evaluative stance (value judgments, importance)
   - Dialogic positioning (engagement with other voices)
   - Alignment with or distancing from propositions

4. **Contextual Sensitivity**:
   - Genre-specific conventions
   - Register appropriateness
   - Cultural and social positioning markers

For each identified feature, provide:
- Exact text span
- Feature category and subcategory
- Functional analysis (what it accomplishes)
- Confidence score (0.0-1.0)
- Theoretical justification

Output format:
```json
{{
  "text_id": "{text_id}",
  "socio_pragmatic_features": [
    {{
      "span": "exact text",
      "start_char": 0,
      "end_char": 10,
      "category": "main_category",
      "subcategory": "specific_type",
      "function": "what this accomplishes pragmatically",
      "confidence": 0.95,
      "explanation": "theoretical justification and context"
    }}
  ],
  "overall_assessment": {{
    "interpersonal_style": "description",
    "authority_level": "high/medium/low",
    "social_distance": "close/neutral/distant",
    "face_orientation": "positive/negative/mixed"
  }}
}}
```"""

    # Academic stance analysis prompt
    ACADEMIC_STANCE_PROMPT = """Analyze the academic stance-taking strategies in the following text, focusing on how the author positions themselves epistemically and evaluatively within scholarly discourse.

Text to analyze:
{text}

Analyze for these stance dimensions:

1. **Epistemic Stance** (Knowledge and Certainty):
   - Certainty markers (definite claims, strong modals)
   - Uncertainty/hedging markers (tentative language, qualifiers)
   - Evidence presentation strategies
   - Knowledge source attributions
   - Epistemic modality markers

2. **Evaluative Stance** (Value and Importance):
   - Significance markers (important, crucial, significant)
   - Value judgments (positive/negative assessments)
   - Novelty claims (new, innovative, unprecedented)
   - Problem identification and gap statements
   - Achievement and contribution claims

3. **Dialogic Stance** (Engagement with Field):
   - Citation integration strategies
   - Agreement/disagreement markers
   - Building on vs. challenging prior work
   - Positioning relative to disciplinary norms
   - Voice management (monoglossic vs. heteroglossic)

4. **Authorial Presence**:
   - Self-mention strategies (I, we, the author)
   - Visibility vs. invisibility choices
   - Authority construction techniques
   - Responsibility claims and disclaimers

Provide detailed analysis with confidence scores and theoretical grounding in academic writing research.

Output as structured JSON with spans, categories, functions, and confidence scores."""

    # Rapport building analysis prompt
    RAPPORT_BUILDING_PROMPT = """Identify and analyze rapport-building strategies in the following text, focusing on how the author establishes and maintains positive relationships with their audience.

Text to analyze:
{text}

Analyze for rapport-building strategies:

1. **Solidarity Markers**:
   - Inclusive pronouns (we, us, our)
   - Shared knowledge assumptions
   - Common ground establishment
   - In-group membership signals

2. **Engagement Strategies**:
   - Direct address to reader
   - Rhetorical questions
   - Interactive elements
   - Attention-maintaining devices

3. **Positive Face Work**:
   - Compliments and appreciation
   - Recognition of reader expertise
   - Validation of reader perspectives
   - Shared value expressions

4. **Relational Positioning**:
   - Equality vs. hierarchy markers
   - Collaboration vs. instruction
   - Empathy and understanding signals
   - Trust-building elements

5. **Linguistic Choices**:
   - Register appropriateness
   - Tone management
   - Formality level
   - Cultural sensitivity markers

For each strategy, identify:
- Specific linguistic realizations
- Functional purpose
- Effectiveness assessment
- Contextual appropriateness

Output detailed JSON analysis with confidence scores."""

    # Hedging and boosting analysis prompt
    HEDGING_BOOSTING_PROMPT = """Analyze hedging and boosting strategies in the following text, examining how the author manages commitment to propositions and modulates the force of their claims.

Text to analyze:
{text}

Focus on:

1. **Hedging Strategies** (Reducing commitment):
   - Modal verbs (might, could, may, should)
   - Epistemic adverbs (possibly, probably, perhaps)
   - Approximators (about, around, roughly)
   - Probability expressions (it is likely that)
   - Conditional constructions
   - Impersonal constructions
   - Passive voice for distancing

2. **Boosting Strategies** (Increasing commitment):
   - Strong modals (must, will, certainly)
   - Emphatic adverbs (clearly, obviously, definitely)
   - Intensifiers (very, extremely, highly)
   - Categorical assertions
   - Factual presentations
   - Authoritative voice markers

3. **Functional Analysis**:
   - Face-saving vs. face-threatening motivations
   - Precision vs. diplomacy trade-offs
   - Authority construction vs. humility
   - Risk management strategies
   - Audience consideration factors

4. **Contextual Factors**:
   - Genre conventions
   - Disciplinary norms
   - Claim types (factual vs. interpretive)
   - Evidence strength
   - Controversial content

Provide comprehensive analysis with theoretical grounding in hedge/boost research."""

    POLITENESS_STRATEGIES_PROMPT = """Analyze politeness strategies in the following text using Brown & Levinson's politeness theory framework, focusing on face-work and social relationship management.

Text to analyze:
{text}

Analyze for:

1. **Positive Politeness Strategies**:
   - Notice and attend to addressee's interests
   - Exaggerate interest, approval, sympathy
   - Use in-group identity markers
   - Seek agreement and avoid disagreement
   - Assert common ground
   - Joke and use humor appropriately
   - Include both speaker and hearer in activity

2. **Negative Politeness Strategies**:
   - Be conventionally indirect
   - Use hedges and qualifiers
   - Apologize and show deference
   - Minimize imposition
   - Give deference to addressee
   - State general rules and avoid assumptions

3. **Off-Record Strategies**:
   - Use metaphors and irony
   - Be vague or ambiguous
   - Overgeneralize
   - Use rhetorical questions
   - Be incomplete or use ellipsis

4. **Bald On-Record** (when appropriate):
   - Direct imperatives in appropriate contexts
   - Emergency or urgency markers
   - Task-oriented directness

Output detailed JSON analysis with strategy identification and effectiveness assessment."""

    CITATION_ANALYSIS_PROMPT = """Analyze citation patterns and academic discourse markers in the following text to understand knowledge construction and authority claims.

Text to analyze:
{text}

Analyze for:

1. **Citation Types and Functions**:
   - Integral citations (author as subject)
   - Non-integral citations (author in parentheses)
   - Multiple citations and citation clusters
   - Self-citations vs. external citations
   - Recent vs. established citations

2. **Citation Purposes**:
   - Attribution of ideas or findings
   - Support for claims and arguments
   - Comparison and contrast
   - Criticism or challenge
   - Gap identification
   - Methodological justification

3. **Knowledge Construction Patterns**:
   - Consensus building ("widely accepted")
   - Knowledge gaps ("however, little is known")
   - Controversy indicators ("debate continues")
   - Novelty claims ("for the first time")
   - Building on previous work

4. **Authority and Credibility Markers**:
   - Use of prominent researchers' names
   - Institutional affiliations mentioned
   - Landmark studies and foundational work
   - Recent developments and cutting-edge research

Provide detailed analysis of how citations construct academic authority and knowledge claims."""

    @classmethod
    def get_prompt_template(cls, prompt_type: PromptType) -> ChatPromptTemplate:
        """Get a specific prompt template by type."""
        system_message = SystemMessagePromptTemplate.from_template(cls.ANNOTATION_SYSTEM_PROMPT)
        
        human_templates = {
            PromptType.SOCIO_PRAGMATIC: cls.SOCIO_PRAGMATIC_PROMPT,
            PromptType.ACADEMIC_STANCE: cls.ACADEMIC_STANCE_PROMPT,
            PromptType.RAPPORT_BUILDING: cls.RAPPORT_BUILDING_PROMPT,
            PromptType.HEDGING_BOOSTING: cls.HEDGING_BOOSTING_PROMPT,
            PromptType.POLITENESS_STRATEGIES: cls.POLITENESS_STRATEGIES_PROMPT,
        }
        
        if prompt_type not in human_templates:
            raise ValueError(f"Prompt type {prompt_type} not implemented")
        
        human_message = HumanMessagePromptTemplate.from_template(human_templates[prompt_type])
        
        return ChatPromptTemplate.from_messages([system_message, human_message])

    @classmethod
    def create_chain_prompt(cls, prompt_types: List[PromptType]) -> ChatPromptTemplate:
        """Create a multi-step prompt chain for comprehensive analysis."""
        system_message = SystemMessagePromptTemplate.from_template(cls.ANNOTATION_SYSTEM_PROMPT)
        
        combined_prompt = """Perform a comprehensive linguistic analysis of the following text using multiple analytical frameworks. 

Text to analyze:
{text}

Perform analysis for the following dimensions:
"""
        
        for i, prompt_type in enumerate(prompt_types, 1):
            if prompt_type == PromptType.SOCIO_PRAGMATIC:
                combined_prompt += f"\n{i}. **Socio-Pragmatic Analysis**: Analyze interpersonal positioning, face work, stance-taking, and contextual sensitivity."
            elif prompt_type == PromptType.ACADEMIC_STANCE:
                combined_prompt += f"\n{i}. **Academic Stance Analysis**: Examine epistemic, evaluative, and dialogic stance strategies."
            elif prompt_type == PromptType.RAPPORT_BUILDING:
                combined_prompt += f"\n{i}. **Rapport Building Analysis**: Identify solidarity markers, engagement strategies, and positive face work."
            elif prompt_type == PromptType.HEDGING_BOOSTING:
                combined_prompt += f"\n{i}. **Hedging/Boosting Analysis**: Analyze commitment modulation and claim strength strategies."
            elif prompt_type == PromptType.POLITENESS_STRATEGIES:
                combined_prompt += f"\n{i}. **Politeness Strategies**: Apply Brown & Levinson's framework to identify face-work strategies."
        
        combined_prompt += """

Provide a comprehensive JSON output that integrates all analyses with:
- Individual feature annotations for each dimension
- Cross-dimensional patterns and interactions
- Overall linguistic profile
- Confidence scores and explanations
- Theoretical grounding for all annotations

Structure your output as a comprehensive annotation object with clear sections for each analytical dimension."""
        
        human_message = HumanMessagePromptTemplate.from_template(combined_prompt)
        return ChatPromptTemplate.from_messages([system_message, human_message])


class PromptChain:
    """Advanced prompt chain for multi-step linguistic annotation."""
    
    def __init__(self, prompt_types: List[PromptType]):
        """Initialize with a list of prompt types to chain together."""
        self.prompt_types = prompt_types
        self.templates = []
        self._build_chain()
    
    def _build_chain(self):
        """Build the prompt template chain."""
        for prompt_type in self.prompt_types:
            template = AdvancedPromptTemplates.get_prompt_template(prompt_type)
            self.templates.append(template)
    
    def get_combined_template(self) -> ChatPromptTemplate:
        """Get a single combined template for all analyses."""
        return AdvancedPromptTemplates.create_chain_prompt(self.prompt_types)
    
    def get_individual_templates(self) -> List[ChatPromptTemplate]:
        """Get individual templates for step-by-step processing."""
        return self.templates


# Predefined prompt configurations
COMPREHENSIVE_ANALYSIS_CHAIN = PromptChain([
    PromptType.SOCIO_PRAGMATIC,
    PromptType.ACADEMIC_STANCE,
    PromptType.RAPPORT_BUILDING,
    PromptType.HEDGING_BOOSTING,
    PromptType.POLITENESS_STRATEGIES
])

ACADEMIC_FOCUS_CHAIN = PromptChain([
    PromptType.ACADEMIC_STANCE,
    PromptType.HEDGING_BOOSTING,
    PromptType.CITATION_ANALYSIS
])

INTERPERSONAL_FOCUS_CHAIN = PromptChain([
    PromptType.SOCIO_PRAGMATIC,
    PromptType.RAPPORT_BUILDING,
    PromptType.POLITENESS_STRATEGIES
])
