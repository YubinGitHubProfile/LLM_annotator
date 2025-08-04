"""
Advanced spaCy-based linguistic analysis for socio-pragmatic features.
Provides sophisticated text analysis capabilities for academic and discourse annotation.
"""

import spacy
from spacy import displacy
from spacy.tokens import Doc, Span, Token
from spacy.matcher import Matcher, PhraseMatcher
from spacy.util import filter_spans
import re
import textstat
from typing import Dict, List, Tuple, Optional, Any, Set, Union
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging
from rapidfuzz import fuzz

from config.settings import SpacyConfig


logger = logging.getLogger(__name__)


class StanceType(Enum):
    """Types of academic stance markers."""
    EPISTEMIC = "epistemic"  # certainty, probability, evidence
    EVALUATIVE = "evaluative"  # importance, desirability, expectedness
    DIALOGIC = "dialogic"  # engagement with other voices


class PolitenessStrategy(Enum):
    """Brown & Levinson politeness strategies."""
    BALD_ON_RECORD = "bald_on_record"
    POSITIVE_POLITENESS = "positive_politeness"
    NEGATIVE_POLITENESS = "negative_politeness"
    OFF_RECORD = "off_record"


@dataclass
class LinguisticFeature:
    """Represents a detected linguistic feature."""
    feature_type: str
    text: str
    start_char: int
    end_char: int
    token_start: int
    token_end: int
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Results of linguistic analysis."""
    text: str
    features: List[LinguisticFeature] = field(default_factory=list)
    readability_scores: Dict[str, float] = field(default_factory=dict)
    discourse_structure: Dict[str, Any] = field(default_factory=dict)
    stance_markers: List[LinguisticFeature] = field(default_factory=list)
    politeness_features: List[LinguisticFeature] = field(default_factory=list)
    rapport_building: List[LinguisticFeature] = field(default_factory=list)
    hedging_boosting: List[LinguisticFeature] = field(default_factory=list)


class AdvancedNLPAnalyzer:
    """Advanced NLP analyzer using spaCy for sophisticated linguistic analysis."""
    
    def __init__(self, config: SpacyConfig):
        """Initialize the analyzer with spaCy models and custom components."""
        self.config = config
        self.nlp = None
        self.matcher = None
        self.phrase_matcher = None
        self._load_models()
        self._setup_custom_patterns()
    
    def _load_models(self):
        """Load spaCy model and initialize components."""
        try:
            self.nlp = spacy.load(self.config.model_name)
            logger.info(f"Loaded spaCy model: {self.config.model_name}")
        except OSError:
            logger.warning(f"Model {self.config.model_name} not found. Downloading...")
            spacy.cli.download(self.config.model_name)
            self.nlp = spacy.load(self.config.model_name)
        
        # Initialize matchers
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Add custom attributes
        if not Token.has_extension("stance_type"):
            Token.set_extension("stance_type", default=None)
        if not Token.has_extension("politeness_strategy"):
            Token.set_extension("politeness_strategy", default=None)
        if not Span.has_extension("discourse_function"):
            Span.set_extension("discourse_function", default=None)
    
    def _setup_custom_patterns(self):
        """Setup custom patterns for linguistic feature detection."""
        # Hedging patterns
        hedging_patterns = [
            # Modal verbs indicating uncertainty
            [{"LOWER": {"IN": ["might", "may", "could", "would", "should"]}}],
            # Epistemic adverbs
            [{"LOWER": {"IN": ["possibly", "probably", "perhaps", "apparently", "seemingly", "arguably"]}}],
            # Hedging phrases
            [{"LOWER": "it"}, {"LOWER": "seems"}, {"LOWER": "that"}],
            [{"LOWER": "it"}, {"LOWER": "appears"}, {"LOWER": "that"}],
            [{"LOWER": "to"}, {"LOWER": "some"}, {"LOWER": "extent"}],
            [{"LOWER": "in"}, {"LOWER": "general"}],
            [{"LOWER": "more"}, {"LOWER": "or"}, {"LOWER": "less"}],
        ]
        
        # Boosting patterns
        boosting_patterns = [
            # Strong modals
            [{"LOWER": {"IN": ["must", "will", "shall", "certainly", "definitely", "undoubtedly"]}}],
            # Intensifiers
            [{"LOWER": {"IN": ["very", "extremely", "highly", "strongly", "clearly", "obviously"]}}],
            # Emphatic phrases
            [{"LOWER": "it"}, {"LOWER": "is"}, {"LOWER": "clear"}, {"LOWER": "that"}],
            [{"LOWER": "without"}, {"LOWER": "doubt"}],
        ]
        
        # Stance markers
        epistemic_stance_patterns = [
            # Evidence markers
            [{"LOWER": {"IN": ["evidence", "data", "research"]}}, {"LOWER": {"IN": ["shows", "indicates", "suggests", "demonstrates"]}}],
            # Knowledge claims
            [{"LOWER": "we"}, {"LOWER": {"IN": ["know", "understand", "recognize"]}}],
            [{"LOWER": "it"}, {"LOWER": "is"}, {"LOWER": {"IN": ["known", "established", "accepted"]}}],
        ]
        
        # Evaluative stance patterns
        evaluative_stance_patterns = [
            # Importance markers
            [{"LOWER": {"IN": ["important", "significant", "crucial", "essential", "vital"]}}],
            # Value judgments
            [{"LOWER": {"IN": ["excellent", "outstanding", "remarkable", "problematic", "concerning"]}}],
        ]
        
        # Rapport building patterns
        rapport_patterns = [
            # Inclusive pronouns
            [{"LOWER": {"IN": ["we", "us", "our"]}}],
            # Shared understanding
            [{"LOWER": "as"}, {"LOWER": "we"}, {"LOWER": {"IN": ["know", "see", "understand"]}}],
            [{"LOWER": "let"}, {"LOWER": "us"}],
            # Engagement markers
            [{"LOWER": {"IN": ["note", "observe", "consider"]}}],
        ]
        
        # Politeness markers
        negative_politeness_patterns = [
            # Apologetic language
            [{"LOWER": {"IN": ["sorry", "apologize", "excuse"]}}],
            # Minimizing impositions
            [{"LOWER": "if"}, {"LOWER": "you"}, {"LOWER": {"IN": ["don't", "wouldn't"]}, "OP": "?"}, {"LOWER": "mind"}],
            [{"LOWER": "would"}, {"LOWER": "you"}, {"LOWER": {"IN": ["mind", "please"]}}],
        ]
        
        # Add patterns to matcher
        self.matcher.add("HEDGING", hedging_patterns)
        self.matcher.add("BOOSTING", boosting_patterns)
        self.matcher.add("EPISTEMIC_STANCE", epistemic_stance_patterns)
        self.matcher.add("EVALUATIVE_STANCE", evaluative_stance_patterns)
        self.matcher.add("RAPPORT_BUILDING", rapport_patterns)
        self.matcher.add("NEGATIVE_POLITENESS", negative_politeness_patterns)
        
        # Discourse markers phrases
        discourse_markers = [
            "however", "therefore", "furthermore", "moreover", "nevertheless",
            "in addition", "on the other hand", "in contrast", "as a result",
            "in conclusion", "to summarize", "first", "second", "finally",
            "for example", "for instance", "such as", "in particular"
        ]
        
        discourse_patterns = [self.nlp(marker) for marker in discourse_markers]
        self.phrase_matcher.add("DISCOURSE_MARKERS", discourse_patterns)
    
    def analyze_text(self, text: str) -> AnalysisResult:
        """Perform comprehensive linguistic analysis on text."""
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Initialize result
        result = AnalysisResult(text=text)
        
        # Extract various linguistic features
        result.features = self._extract_all_features(doc)
        result.readability_scores = self._calculate_readability_scores(text)
        result.discourse_structure = self._analyze_discourse_structure(doc)
        
        # Categorize features
        result.stance_markers = [f for f in result.features if "stance" in f.feature_type.lower()]
        result.politeness_features = [f for f in result.features if "politeness" in f.feature_type.lower()]
        result.rapport_building = [f for f in result.features if "rapport" in f.feature_type.lower()]
        result.hedging_boosting = [f for f in result.features if f.feature_type.lower() in ["hedging", "boosting"]]
        
        return result
    
    def _extract_all_features(self, doc: Doc) -> List[LinguisticFeature]:
        """Extract all linguistic features from document."""
        features = []
        
        # Pattern-based features
        matches = self.matcher(doc)
        phrase_matches = self.phrase_matcher(doc)
        
        # Process matcher results
        for match_id, start, end in matches:
            span = doc[start:end]
            match_label = self.nlp.vocab.strings[match_id]
            
            feature = LinguisticFeature(
                feature_type=match_label.lower(),
                text=span.text,
                start_char=span.start_char,
                end_char=span.end_char,
                token_start=start,
                token_end=end,
                properties={"pattern_based": True}
            )
            features.append(feature)
        
        # Process phrase matcher results
        for match_id, start, end in phrase_matches:
            span = doc[start:end]
            match_label = self.nlp.vocab.strings[match_id]
            
            feature = LinguisticFeature(
                feature_type=match_label.lower(),
                text=span.text,
                start_char=span.start_char,
                end_char=span.end_char,
                token_start=start,
                token_end=end,
                properties={"phrase_based": True}
            )
            features.append(feature)
        
        # Add grammatical features
        features.extend(self._extract_grammatical_features(doc))
        
        # Add semantic features
        features.extend(self._extract_semantic_features(doc))
        
        return features
    
    def _extract_grammatical_features(self, doc: Doc) -> List[LinguisticFeature]:
        """Extract grammatical features relevant to socio-pragmatic analysis."""
        features = []
        
        for sent in doc.sents:
            # Passive voice detection
            if self._is_passive_voice(sent):
                features.append(LinguisticFeature(
                    feature_type="passive_voice",
                    text=sent.text,
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                    token_start=sent.start,
                    token_end=sent.end,
                    properties={"grammatical": True, "sentence_level": True}
                ))
            
            # Nominalization detection
            nominalizations = self._detect_nominalizations(sent)
            for nom in nominalizations:
                features.append(LinguisticFeature(
                    feature_type="nominalization",
                    text=nom.text,
                    start_char=nom.idx,
                    end_char=nom.idx + len(nom.text),
                    token_start=nom.i,
                    token_end=nom.i + 1,
                    properties={"grammatical": True, "pos": nom.pos_}
                ))
        
        return features
    
    def _extract_semantic_features(self, doc: Doc) -> List[LinguisticFeature]:
        """Extract semantic features for advanced analysis."""
        features = []
        
        # Extract named entities with academic relevance
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "WORK_OF_ART", "EVENT"]:
                features.append(LinguisticFeature(
                    feature_type=f"entity_{ent.label_.lower()}",
                    text=ent.text,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    token_start=ent.start,
                    token_end=ent.end,
                    properties={"semantic": True, "entity_type": ent.label_}
                ))
        
        # Extract academic citations patterns
        citation_patterns = self._detect_citations(doc)
        features.extend(citation_patterns)
        
        return features
    
    def _is_passive_voice(self, span: Span) -> bool:
        """Detect passive voice constructions."""
        # Look for auxiliary verb + past participle pattern
        for token in span:
            if (token.dep_ == "auxpass" or 
                (token.lemma_ in ["be", "get"] and 
                 token.head.tag_ in ["VBN", "VBD"] and 
                 token.head.dep_ in ["ROOT", "ccomp", "xcomp"])):
                return True
        return False
    
    def _detect_nominalizations(self, span: Span) -> List[Token]:
        """Detect nominalization patterns."""
        nominalizations = []
        nominalization_suffixes = ["-tion", "-sion", "-ment", "-ness", "-ity", "-ity", "-ance", "-ence"]
        
        for token in span:
            if (token.pos_ == "NOUN" and 
                any(token.text.lower().endswith(suffix.replace("-", "")) for suffix in nominalization_suffixes)):
                nominalizations.append(token)
        
        return nominalizations
    
    def _detect_citations(self, doc: Doc) -> List[LinguisticFeature]:
        """Detect academic citation patterns."""
        features = []
        
        # Pattern for author-date citations
        citation_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)*\s*\(\d{4}[a-z]?\)')
        
        for match in citation_pattern.finditer(doc.text):
            features.append(LinguisticFeature(
                feature_type="academic_citation",
                text=match.group(),
                start_char=match.start(),
                end_char=match.end(),
                token_start=-1,  # Will be calculated if needed
                token_end=-1,
                properties={"citation_type": "author_date"}
            ))
        
        return features
    
    def _calculate_readability_scores(self, text: str) -> Dict[str, float]:
        """Calculate various readability metrics."""
        try:
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "gunning_fog": textstat.gunning_fog(text),
                "automated_readability_index": textstat.automated_readability_index(text),
                "coleman_liau_index": textstat.coleman_liau_index(text),
                "difficult_words": textstat.difficult_words(text),
                "linsear_write_formula": textstat.linsear_write_formula(text),
                "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
            }
        except Exception as e:
            logger.warning(f"Error calculating readability scores: {e}")
            return {}
    
    def _analyze_discourse_structure(self, doc: Doc) -> Dict[str, Any]:
        """Analyze discourse structure and organization."""
        structure = {
            "sentence_count": len(list(doc.sents)),
            "paragraph_count": len(doc.text.split("\n\n")),
            "avg_sentence_length": np.mean([len(sent) for sent in doc.sents]) if doc.sents else 0,
            "discourse_markers": [],
            "transition_signals": [],
        }
        
        # Find discourse markers and their positions
        matches = self.phrase_matcher(doc)
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "DISCOURSE_MARKERS":
                span = doc[start:end]
                structure["discourse_markers"].append({
                    "text": span.text,
                    "position": "initial" if start < 3 else "medial" if end < len(doc) - 3 else "final",
                    "sentence_position": self._get_sentence_position(span),
                })
        
        return structure
    
    def _get_sentence_position(self, span: Span) -> str:
        """Determine position of span within its sentence."""
        sent = span.sent
        relative_pos = (span.start - sent.start) / len(sent)
        
        if relative_pos < 0.33:
            return "initial"
        elif relative_pos < 0.67:
            return "medial"
        else:
            return "final"
    
    def visualize_features(self, analysis_result: AnalysisResult) -> str:
        """Generate HTML visualization of linguistic features."""
        doc = self.nlp(analysis_result.text)
        
        # Create custom entities for visualization
        ents = []
        for feature in analysis_result.features:
            if feature.token_start >= 0 and feature.token_end >= 0:
                ents.append(Span(doc, feature.token_start, feature.token_end, label=feature.feature_type.upper()))
        
        # Filter overlapping spans
        doc.ents = filter_spans(ents)
        
        # Generate HTML
        html = displacy.render(doc, style="ent", jupyter=False)
        return html
    
    def get_feature_statistics(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Generate statistics about detected features."""
        stats = {
            "total_features": len(analysis_result.features),
            "feature_types": {},
            "density": len(analysis_result.features) / len(analysis_result.text.split()) if analysis_result.text else 0,
        }
        
        # Count features by type
        for feature in analysis_result.features:
            feature_type = feature.feature_type
            if feature_type not in stats["feature_types"]:
                stats["feature_types"][feature_type] = 0
            stats["feature_types"][feature_type] += 1
        
        # Add readability statistics
        stats["readability"] = analysis_result.readability_scores
        
        return stats
    
    def classify_by_keywords(self, text: str, keywords: List[Union[str, Tuple[Union[str, Tuple[str, str]], ...]]], 
                           feature_name: str = "keyword_match", 
                           fuzzy_threshold: float = 0.8,
                           enable_fuzzy: bool = True) -> Dict[str, Any]:
        """
        Advanced binary classification with fuzzy matching and tuple-based AND logic.
        
        Args:
            text: Input text to classify
            keywords: List of keywords/tuples. Can contain:
                     - Simple strings: "important"
                     - Tuples for AND logic: ("research", "shows")
                     - Tuples with POS tags: (("research", "NOUN"), ("shows", "VERB"))
                     - Mixed: ["simple", ("complex", "tuple"), (("word", "NOUN"), "another")]
            feature_name: Name for the feature being detected
            fuzzy_threshold: Similarity threshold for fuzzy matching (0.0-1.0)
            enable_fuzzy: Whether to enable fuzzy matching for words >=5 chars
            
        Returns:
            Dictionary with text, label (1/0), matched items, and details
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        matched_items = []
        match_details = []
        
        for keyword_item in keywords:
            if isinstance(keyword_item, str):
                # Simple string matching
                match_result = self._match_simple_keyword(doc, keyword_item, fuzzy_threshold, enable_fuzzy)
                if match_result:
                    matched_items.append(keyword_item)
                    match_details.append({
                        "type": "simple",
                        "keyword": keyword_item,
                        "matches": match_result
                    })
                    
            elif isinstance(keyword_item, tuple):
                # Tuple matching (AND logic)
                match_result = self._match_tuple_keyword(doc, keyword_item, fuzzy_threshold, enable_fuzzy)
                if match_result:
                    matched_items.append(keyword_item)
                    match_details.append({
                        "type": "tuple",
                        "keyword": keyword_item,
                        "matches": match_result
                    })
        
        # Binary classification: 1 if any keyword/tuple matched, 0 otherwise
        label = 1 if matched_items else 0
        
        return {
            "text": text,
            "label": label,
            "feature_name": feature_name,
            "matched_items": matched_items,
            "match_details": match_details,
            "total_matches": len(matched_items),
            "fuzzy_enabled": enable_fuzzy,
            "fuzzy_threshold": fuzzy_threshold
        }
    
    def _match_simple_keyword(self, doc: Doc, keyword: str, fuzzy_threshold: float, enable_fuzzy: bool) -> List[Dict[str, Any]]:
        """Match a simple string keyword with optional fuzzy matching."""
        matches = []
        keyword_lower = keyword.lower()
        
        # Check for exact matches first
        for token in doc:
            if token.text.lower() == keyword_lower:
                matches.append({
                    "text": token.text,
                    "match_type": "exact",
                    "similarity": 1.0,
                    "start": token.idx,
                    "end": token.idx + len(token.text)
                })
        
        # Fuzzy matching for words >= 5 characters
        if enable_fuzzy and len(keyword) >= 5 and not matches:
            for token in doc:
                if len(token.text) >= 5:
                    # Use rapidfuzz for better fuzzy matching (returns 0-100, normalize to 0-1)
                    similarity = fuzz.ratio(token.text.lower(), keyword_lower) / 100.0
                    if similarity >= fuzzy_threshold:
                        matches.append({
                            "text": token.text,
                            "match_type": "fuzzy",
                            "similarity": similarity,
                            "start": token.idx,
                            "end": token.idx + len(token.text)
                        })
        
        # Also check for multi-word phrases
        if ' ' in keyword:
            text_lower = doc.text.lower()
            if keyword_lower in text_lower:
                start_pos = text_lower.find(keyword_lower)
                matches.append({
                    "text": keyword,
                    "match_type": "phrase",
                    "similarity": 1.0,
                    "start": start_pos,
                    "end": start_pos + len(keyword)
                })
        
        return matches
    
    def _match_tuple_keyword(self, doc: Doc, keyword_tuple: Tuple, fuzzy_threshold: float, enable_fuzzy: bool) -> List[Dict[str, Any]]:
        """Match a tuple of keywords/POS combinations using AND logic."""
        all_element_matches = []
        
        # Process each element in the tuple
        for element in keyword_tuple:
            element_matches = []
            
            if isinstance(element, str):
                # Simple string element
                element_matches = self._match_simple_keyword(doc, element, fuzzy_threshold, enable_fuzzy)
                
            elif isinstance(element, tuple) and len(element) == 2:
                # (word, POS) tuple
                word, pos_tag = element
                word_lower = word.lower()
                
                # Find tokens that match both word and POS
                for token in doc:
                    word_match = False
                    
                    # Check exact match
                    if token.text.lower() == word_lower:
                        word_match = True
                        similarity = 1.0
                        match_type = "exact"
                    
                    # Check fuzzy match for words >= 5 chars
                    elif enable_fuzzy and len(word) >= 5 and len(token.text) >= 5:
                        # Use rapidfuzz for better fuzzy matching (returns 0-100, normalize to 0-1)
                        similarity = fuzz.ratio(token.text.lower(), word_lower) / 100.0
                        if similarity >= fuzzy_threshold:
                            word_match = True
                            match_type = "fuzzy"
                    
                    # Check POS tag match
                    pos_match = (token.pos_ == pos_tag or token.tag_ == pos_tag)
                    
                    if word_match and pos_match:
                        element_matches.append({
                            "text": token.text,
                            "pos": token.pos_,
                            "tag": token.tag_,
                            "match_type": match_type,
                            "similarity": similarity,
                            "start": token.idx,
                            "end": token.idx + len(token.text),
                            "required_pos": pos_tag
                        })
            
            all_element_matches.append(element_matches)
        
        # Check if ALL elements in the tuple have matches (AND logic)
        if all(matches for matches in all_element_matches):
            # Combine all matches for this tuple
            combined_matches = []
            for i, element_matches in enumerate(all_element_matches):
                combined_matches.extend([{
                    **match,
                    "element_index": i,
                    "element": keyword_tuple[i]
                } for match in element_matches])
            
            return combined_matches
        
        return []  # Not all elements matched
    
    def batch_classify_by_keywords(self, texts: List[str], 
                                 keywords: List[Union[str, Tuple[Union[str, Tuple[str, str]], ...]]], 
                                 feature_name: str = "keyword_match",
                                 fuzzy_threshold: float = 0.8,
                                 enable_fuzzy: bool = True) -> List[Dict[str, Any]]:
        """
        Classify multiple texts using advanced keyword matching.
        
        Args:
            texts: List of texts to classify
            keywords: List of keywords/tuples (same format as classify_by_keywords)
            feature_name: Name for the feature being detected
            fuzzy_threshold: Similarity threshold for fuzzy matching
            enable_fuzzy: Whether to enable fuzzy matching
            
        Returns:
            List of classification results
        """
        results = []
        for text in texts:
            result = self.classify_by_keywords(text, keywords, feature_name, fuzzy_threshold, enable_fuzzy)
            results.append(result)
        
        return results
