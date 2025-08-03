"""
Advanced data loader for sophisticated document annotation.
Supports multiple input formats and batch processing with enhanced features.
"""

import pandas as pd
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator
from dataclasses import dataclass
import time
import hashlib

from config.settings import Config, AnnotationType
from agents.advanced_annotator import AdvancedAnnotator


logger = logging.getLogger(__name__)


@dataclass
class DocumentBatch:
    """Represents a batch of documents for processing."""
    texts: List[str]
    text_ids: List[str]
    metadata: List[Dict[str, Any]]
    batch_id: str
    batch_size: int


@dataclass
class ProcessingResult:
    """Results from processing a batch of documents."""
    batch_id: str
    successful_count: int
    failed_count: int
    results: List[Dict[str, Any]]
    processing_time: float
    errors: List[str]


class AdvancedDataLoader:
    """Advanced data loader with support for multiple formats and sophisticated processing."""
    
    def __init__(self, config: Config, annotator: AdvancedAnnotator):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration object
            annotator: Advanced annotator instance
        """
        self.config = config
        self.annotator = annotator
        self.batch_size = config.annotation_config.batch_size
        
        # Processing statistics
        self.stats = {
            "total_documents": 0,
            "total_batches": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_processing_time": 0.0,
            "average_batch_time": 0.0,
        }
    
    def _detect_file_format(self, filename: str) -> str:
        """Detect file format based on extension."""
        path = Path(filename)
        extension = path.suffix.lower()
        
        if extension == '.csv':
            return 'csv'
        elif extension == '.json':
            return 'json'
        elif extension in ['.txt', '.text']:
            return 'txt'
        else:
            return 'unknown'
    
    def load_from_csv(
        self, 
        file_path: Union[str, Path], 
        text_column: str = "text",
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None
    ) -> Iterator[DocumentBatch]:
        """
        Load documents from CSV file in batches.
        
        Args:
            file_path: Path to CSV file
            text_column: Name of column containing text
            id_column: Name of column containing document IDs (optional)
            metadata_columns: List of columns to include as metadata
        
        Yields:
            DocumentBatch objects
        """
        df = pd.read_csv(file_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV")
        
        # Prepare data
        texts = df[text_column].astype(str).tolist()
        
        if id_column and id_column in df.columns:
            text_ids = df[id_column].astype(str).tolist()
        else:
            text_ids = [f"doc_{i}" for i in range(len(texts))]
        
        if metadata_columns:
            metadata_cols = [col for col in metadata_columns if col in df.columns]
            metadata = df[metadata_cols].to_dict('records')
        else:
            metadata = [{}] * len(texts)
        
        # Generate batches
        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            
            batch = DocumentBatch(
                texts=texts[batch_start:batch_end],
                text_ids=text_ids[batch_start:batch_end],
                metadata=metadata[batch_start:batch_end],
                batch_id=f"csv_batch_{batch_start//self.batch_size}",
                batch_size=batch_end - batch_start
            )
            
            yield batch
    
    def load_from_json(
        self, 
        file_path: Union[str, Path],
        text_field: str = "text",
        id_field: Optional[str] = None
    ) -> Iterator[DocumentBatch]:
        """
        Load documents from JSON file in batches.
        
        Args:
            file_path: Path to JSON file
            text_field: Field name containing text
            id_field: Field name containing document ID (optional)
        
        Yields:
            DocumentBatch objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]  # Single document
        
        texts = []
        text_ids = []
        metadata = []
        
        for i, doc in enumerate(data):
            if text_field in doc:
                texts.append(str(doc[text_field]))
                
                if id_field and id_field in doc:
                    text_ids.append(str(doc[id_field]))
                else:
                    text_ids.append(f"doc_{i}")
                
                # Include all other fields as metadata
                meta = {k: v for k, v in doc.items() if k not in [text_field, id_field]}
                metadata.append(meta)
        
        # Generate batches
        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            
            batch = DocumentBatch(
                texts=texts[batch_start:batch_end],
                text_ids=text_ids[batch_start:batch_end],
                metadata=metadata[batch_start:batch_end],
                batch_id=f"json_batch_{batch_start//self.batch_size}",
                batch_size=batch_end - batch_start
            )
            
            yield batch
    
    def load_from_text_files(
        self, 
        directory_path: Union[str, Path],
        file_pattern: str = "*.txt",
        encoding: str = "utf-8"
    ) -> Iterator[DocumentBatch]:
        """
        Load documents from text files in a directory.
        
        Args:
            directory_path: Path to directory containing text files
            file_pattern: Glob pattern for file selection
            encoding: File encoding
        
        Yields:
            DocumentBatch objects
        """
        directory = Path(directory_path)
        text_files = list(directory.glob(file_pattern))
        
        texts = []
        text_ids = []
        metadata = []
        
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                
                texts.append(text)
                text_ids.append(file_path.stem)
                metadata.append({
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "modified_time": file_path.stat().st_mtime
                })
                
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {str(e)}")
                continue
        
        # Generate batches
        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            
            batch = DocumentBatch(
                texts=texts[batch_start:batch_end],
                text_ids=text_ids[batch_start:batch_end],
                metadata=metadata[batch_start:batch_end],
                batch_id=f"files_batch_{batch_start//self.batch_size}",
                batch_size=batch_end - batch_start
            )
            
            yield batch
    
    def process_batch(
        self, 
        batch: DocumentBatch,
        annotation_types: Optional[List[AnnotationType]] = None,
        include_nlp_analysis: bool = True,
        **kwargs
    ) -> ProcessingResult:
        """
        Process a single batch of documents.
        
        Args:
            batch: DocumentBatch to process
            annotation_types: Types of annotations to perform
            include_nlp_analysis: Whether to include spaCy analysis
            **kwargs: Additional parameters for annotation
        
        Returns:
            ProcessingResult object
        """
        start_time = time.time()
        
        logger.info(f"Processing batch {batch.batch_id} with {batch.batch_size} documents")
        
        results = []
        errors = []
        successful_count = 0
        failed_count = 0
        
        for text, text_id, metadata in zip(batch.texts, batch.text_ids, batch.metadata):
            try:
                # Annotate the text
                result = self.annotator.annotate_text(
                    text=text,
                    text_id=text_id,
                    annotation_types=annotation_types,
                    include_nlp_analysis=include_nlp_analysis,
                    **kwargs
                )
                
                # Add batch metadata
                result["metadata"].update({
                    "batch_id": batch.batch_id,
                    "original_metadata": metadata
                })
                
                results.append(result)
                
                if "error" not in result:
                    successful_count += 1
                else:
                    failed_count += 1
                    errors.append(f"Text {text_id}: {result['error']}")
                
            except Exception as e:
                logger.error(f"Error processing text {text_id}: {str(e)}")
                failed_count += 1
                errors.append(f"Text {text_id}: {str(e)}")
                
                # Add error result
                results.append({
                    "text_id": text_id,
                    "error": str(e),
                    "metadata": {
                        "batch_id": batch.batch_id,
                        "original_metadata": metadata
                    }
                })
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats["total_batches"] += 1
        self.stats["total_documents"] += batch.batch_size
        self.stats["successful_documents"] += successful_count
        self.stats["failed_documents"] += failed_count
        self.stats["total_processing_time"] += processing_time
        self.stats["average_batch_time"] = (
            self.stats["total_processing_time"] / self.stats["total_batches"]
        )
        
        return ProcessingResult(
            batch_id=batch.batch_id,
            successful_count=successful_count,
            failed_count=failed_count,
            results=results,
            processing_time=processing_time,
            errors=errors
        )
    
    def annotate_csv(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        text_column: str = "text",
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        annotation_types: Optional[List[AnnotationType]] = None,
        include_nlp_analysis: bool = True,
        save_intermediate: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Annotate an entire CSV file and save results.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output file
            text_column: Name of column containing text
            id_column: Name of column containing document IDs
            metadata_columns: List of columns to include as metadata
            annotation_types: Types of annotations to perform
            include_nlp_analysis: Whether to include spaCy analysis
            save_intermediate: Whether to save intermediate results
            **kwargs: Additional parameters for annotation
        
        Returns:
            Dictionary with processing summary
        """
        start_time = time.time()
        
        all_results = []
        all_errors = []
        total_successful = 0
        total_failed = 0
        
        # Process batches
        for batch in self.load_from_csv(input_file, text_column, id_column, metadata_columns):
            result = self.process_batch(
                batch=batch,
                annotation_types=annotation_types,
                include_nlp_analysis=include_nlp_analysis,
                **kwargs
            )
            
            all_results.extend(result.results)
            all_errors.extend(result.errors)
            total_successful += result.successful_count
            total_failed += result.failed_count
            
            # Save intermediate results if requested
            if save_intermediate and result.successful_count > 0:
                intermediate_file = Path(output_file).with_suffix(f".batch_{batch.batch_id}.json")
                self._save_results_json(result.results, intermediate_file)
        
        # Save final results
        self._save_results_json(all_results, output_file)
        
        # Also save as CSV if requested
        if str(output_file).endswith('.csv'):
            self._save_results_csv(all_results, output_file)
        elif not str(output_file).endswith('.json'):
            # Default to JSON if no extension specified
            json_output = Path(output_file).with_suffix('.json')
            self._save_results_json(all_results, json_output)
        
        total_time = time.time() - start_time
        
        summary = {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "total_documents": len(all_results),
            "successful_annotations": total_successful,
            "failed_annotations": total_failed,
            "total_processing_time": total_time,
            "errors": all_errors,
            "annotation_types": [t.value for t in annotation_types] if annotation_types else [],
            "statistics": self.get_statistics()
        }
        
        logger.info(f"Completed annotation of {len(all_results)} documents in {total_time:.2f} seconds")
        
        return summary
    
    def _save_results_json(self, results: List[Dict[str, Any]], output_file: Union[str, Path]):
        """Save results to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_results_csv(self, results: List[Dict[str, Any]], output_file: Union[str, Path]):
        """Save results to CSV file with flattened structure."""
        if not results:
            return
        
        # Flatten the results for CSV output
        flattened_results = []
        
        for result in results:
            flat_result = {
                "text_id": result.get("text_id", ""),
                "original_text": result.get("original_text", ""),
                "has_error": "error" in result,
                "error_message": result.get("error", ""),
            }
            
            # Add metadata
            metadata = result.get("metadata", {})
            for key, value in metadata.items():
                if key != "original_metadata":
                    flat_result[f"meta_{key}"] = value
            
            # Add original metadata
            original_metadata = metadata.get("original_metadata", {})
            for key, value in original_metadata.items():
                flat_result[f"orig_{key}"] = value
            
            # Add LLM annotations summary
            llm_annotations = result.get("llm_annotations", {})
            for annotation_type, annotation_data in llm_annotations.items():
                if isinstance(annotation_data, dict) and not annotation_data.get("error"):
                    # Count features for each annotation type
                    feature_count = 0
                    for key, value in annotation_data.items():
                        if isinstance(value, list) and key.endswith(('_features', '_markers', '_strategies')):
                            feature_count += len(value)
                    flat_result[f"{annotation_type}_feature_count"] = feature_count
            
            # Add NLP analysis summary
            nlp_analysis = result.get("nlp_analysis")
            if nlp_analysis:
                flat_result["nlp_feature_count"] = len(nlp_analysis.get("features", []))
                readability = nlp_analysis.get("readability_scores", {})
                for key, value in readability.items():
                    flat_result[f"readability_{key}"] = value
            
            flattened_results.append(flat_result)
        
        # Save to CSV
        df = pd.DataFrame(flattened_results)
        df.to_csv(output_file, index=False, encoding='utf-8')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            "total_documents": 0,
            "total_batches": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_processing_time": 0.0,
            "average_batch_time": 0.0,
        }


class DocumentPreprocessor:
    """Preprocessor for cleaning and preparing documents for annotation."""
    
    def __init__(self, config: Config):
        """Initialize preprocessor with configuration."""
        self.config = config
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for annotation.
        
        Args:
            text: Raw text to preprocess
        
        Returns:
            Preprocessed text
        """
        if not self.config.annotation_config.enable_preprocessing:
            return text
        
        # Basic cleaning
        text = self._clean_text(text)
        
        # Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Handle special characters
        text = self._handle_special_characters(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace patterns."""
        import re
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        # Normalize multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Normalize multiple line breaks
        text = re.sub(r'\n+', '\n', text)
        
        return text
    
    def _handle_special_characters(self, text: str) -> str:
        """Handle special characters and encoding issues."""
        # Replace common Unicode characters
        replacements = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '…': '...',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
