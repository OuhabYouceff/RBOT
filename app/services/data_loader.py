"""
Module for loading and preprocessing RNE laws data from multiple JSON files.
"""

import json
import os
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path

class RNEDataLoader:
    """Class for loading and preprocessing RNE laws data from multiple JSON files."""
    
    def __init__(self, data_paths: List[str] = None, data_path: str = None, expected_files: Dict[str, str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_paths: List of specific file paths to load (e.g., ['data/external_data.json', 'data/rne_laws.json'])
            data_path: Path to directory (for backward compatibility) or single file path
            expected_files: Dictionary mapping filenames to descriptions (optional).
        """
        # Handle both new list-based approach and backward compatibility
        if data_paths:
            self.data_paths = data_paths
        elif data_path:
            if isinstance(data_path, list):
                self.data_paths = data_path
            else:
                self.data_paths = [data_path]  # Convert single path to list for consistency
        else:
            raise ValueError("Either data_paths (list) or data_path must be provided")
            
        self.expected_files = expected_files or {}
        self.raw_data = {}  # Will store data by filename
        self.processed_data = None
        
    def get_file_paths(self) -> List[str]:
        """
        Get the list of file paths to load.
        
        Returns:
            List of file paths to process.
        """
        file_paths = []
        
        for path in self.data_paths:
            if os.path.isfile(path):
                # Direct file path
                if path.endswith('.json'):
                    file_paths.append(path)
                else:
                    logging.warning(f"Skipping non-JSON file: {path}")
            elif os.path.isdir(path):
                # Directory path - scan for JSON files
                for filename in os.listdir(path):
                    if filename.endswith('.json'):
                        file_path = os.path.join(path, filename)
                        file_paths.append(file_path)
            else:
                logging.error(f"Path not found: {path}")
                
        logging.info(f"Found {len(file_paths)} JSON files: {[os.path.basename(f) for f in file_paths]}")
        return file_paths
    
    def load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load data from all specified JSON files.
        
        Returns:
            Dictionary mapping filenames to their loaded data.
        """
        json_files = self.get_file_paths()
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found from the provided paths: {self.data_paths}")
        
        self.raw_data = {}
        
        for file_path in json_files:
            filename = os.path.basename(file_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    
                # If the data is a single dictionary, convert it to a list
                if isinstance(file_data, dict):
                    file_data = [file_data]
                
                self.raw_data[filename] = file_data
                
                # Log information about loaded file
                description = self.expected_files.get(filename, "Unknown data type")
                logging.info(f"Successfully loaded {len(file_data)} items from {filename} ({description})")
                
            except json.JSONDecodeError as e:
                error_message = f"Error parsing JSON file {filename}: {str(e)}"
                logging.error(error_message)
                # Continue loading other files instead of failing completely
                continue
            except Exception as e:
                error_message = f"Unexpected error loading data from {filename}: {str(e)}"
                logging.error(error_message)
                continue
        
        # Check if expected files were loaded
        if self.expected_files:
            missing_files = set(self.expected_files.keys()) - set(self.raw_data.keys())
            if missing_files:
                logging.warning(f"Expected files not found: {missing_files}")
        
        total_items = sum(len(data) for data in self.raw_data.values())
        logging.info(f"Total items loaded across all files: {total_items}")
        
        return self.raw_data
    
    def process_data(self) -> List[Dict[str, Any]]:
        """
        Process the raw data from all files into a format suitable for indexing.
        
        Returns:
            List of processed documents for indexing.
        """
        if not self.raw_data:
            self.load_data()
            
        processed_data = []
        
        for filename, file_data in self.raw_data.items():
            file_processed = self._process_file_data(filename, file_data)
            processed_data.extend(file_processed)
        
        self.processed_data = processed_data
        logging.info(f"Processed {len(processed_data)} documents successfully across all files")
        return processed_data
    
    def _process_file_data(self, filename: str, file_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process data from a specific file.
        
        Args:
            filename: Name of the source file.
            file_data: Data loaded from the file.
            
        Returns:
            List of processed documents from this file.
        """
        processed_data = []
        file_prefix = filename.replace('.json', '')
        
        for idx, item in enumerate(file_data):
            try:
                # Handle RNE laws data (with multilingual content)
                if filename == "rne_laws.json" and self._is_rne_law_item(item):
                    processed_data.extend(self._process_rne_law_item(item))
                
                # Handle other data types (external_data.json, fiscal_knowledge.json, etc.)
                else:
                    processed_doc = self._process_general_item(item, filename, idx)
                    if processed_doc:
                        processed_data.append(processed_doc)
                        
            except Exception as e:
                item_id = item.get('code', item.get('id', f'{filename}_{idx}'))
                logging.error(f"Error processing item {item_id} from {filename}: {str(e)}")
                continue
        
        logging.info(f"Processed {len(processed_data)} documents from {filename}")
        return processed_data
    
    def _is_rne_law_item(self, item: Dict[str, Any]) -> bool:
        """
        Check if an item is an RNE law item (has multilingual content).
        
        Args:
            item: Item to check.
            
        Returns:
            True if item appears to be an RNE law item.
        """
        return ("french_content" in item or "arabic_content" in item) and "code" in item
    
    def _process_rne_law_item(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process an RNE law item (handles multilingual content).
        
        Args:
            item: RNE law item to process.
            
        Returns:
            List of processed documents (one per language).
        """
        processed_docs = []
        
        # Process French content if available
        if "french_content" in item and item["french_content"]:
            french_doc = {
                "id": f"{item['code']}_fr",
                "code": item["code"],
                "language": "fr",
                "source_file": "rne_laws.json",
                "data_type": "rne_law",
                "type_entreprise": item.get("type_entreprise", ""),
                "genre_entreprise": item.get("genre_entreprise", ""),
                "procedure": item.get("procedure", ""),
                "redevance_demandee": item.get("redevance_demandee", ""),
                "delais": item.get("delais", ""),
                "pdf_link": item.get("pdf_french_link", ""),
                "content": self._process_content(item["french_content"]),
                "raw_content": item["french_content"]
            }
            processed_docs.append(french_doc)
            
        # Process Arabic content if available
        if "arabic_content" in item and item["arabic_content"]:
            arabic_doc = {
                "id": f"{item['code']}_ar",
                "code": item["code"],
                "language": "ar",
                "source_file": "rne_laws.json",
                "data_type": "rne_law",
                "type_entreprise": item.get("type_entreprise", ""),
                "genre_entreprise": item.get("genre_entreprise", ""),
                "procedure": item.get("procedure", ""),
                "redevance_demandee": item.get("redevance_demandee", ""),
                "delais": item.get("delais", ""),
                "pdf_link": item.get("pdf_arabic_link", ""),
                "content": self._process_content(item["arabic_content"]),
                "raw_content": item["arabic_content"]
            }
            processed_docs.append(arabic_doc)
            
        return processed_docs
    
    def _process_general_item(self, item: Dict[str, Any], filename: str, idx: int) -> Dict[str, Any]:
        """
        Process a general item from non-RNE law files.
        
        Args:
            item: Item to process.
            filename: Source filename.
            idx: Item index in the file.
            
        Returns:
            Processed document dictionary.
        """
        # Generate ID based on available fields
        item_id = item.get('id', item.get('code', f"{filename.replace('.json', '')}_{idx}"))
        
        # Determine data type based on filename
        data_type_map = {
            "external_data.json": "business_fiscal",
            "fiscal_knowledge.json": "fiscal_info",
            "rne_laws.json": "rne_law"
        }
        data_type = data_type_map.get(filename, "general")
        
        # Process content - handle different possible structures
        content = ""
        if isinstance(item, dict):
            # Try to extract meaningful content from the item
            content_fields = []
            for key, value in item.items():
                if key not in ['id', 'code'] and value:  # Skip ID fields and empty values
                    if isinstance(value, (str, int, float)):
                        content_fields.append(f"{key}: {value}")
                    elif isinstance(value, list):
                        list_text = ', '.join(str(v) for v in value if v)
                        if list_text:
                            content_fields.append(f"{key}: {list_text}")
                    elif isinstance(value, dict):
                        dict_text = self._process_content(value)
                        if dict_text:
                            content_fields.append(f"{key}: {dict_text}")
            
            content = '\n'.join(content_fields)
        
        return {
            "id": str(item_id),
            "source_file": filename,
            "data_type": data_type,
            "language": "fr",  # Default to French, could be enhanced to detect language
            "content": content,
            "raw_content": item
        }
    
    def _process_content(self, content: Dict[str, Any]) -> str:
        """
        Process content dictionary into a single string for indexing.
        
        Args:
            content: Dictionary containing content fields.
            
        Returns:
            Processed content as a single string.
        """
        # Handle case where content is None or not a dictionary
        if not content or not isinstance(content, dict):
            return ""
            
        processed_text = ""
        
        try:
            for key, value in content.items():
                if isinstance(value, list):
                    # Join list items and add to processed text
                    list_text = ' '.join(str(item) for item in value if item)
                    if list_text.strip():
                        processed_text += f"{key}: {list_text}\n"
                elif value:  # Only add non-empty values
                    processed_text += f"{key}: {str(value)}\n"
        except Exception as e:
            logging.error(f"Error processing content: {str(e)}")
            return ""
                
        return processed_text.strip()
    
    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID to retrieve.
            
        Returns:
            Document dictionary or None if not found.
        """
        if self.processed_data is None:
            self.process_data()
            
        for doc in self.processed_data:
            if doc["id"] == doc_id:
                return doc
                
        return None
    
    def get_documents_by_source(self, filename: str) -> List[Dict[str, Any]]:
        """
        Retrieve all documents from a specific source file.
        
        Args:
            filename: Source filename to filter by.
            
        Returns:
            List of documents from the specified file.
        """
        if self.processed_data is None:
            self.process_data()
            
        return [doc for doc in self.processed_data if doc.get("source_file") == filename]
    
    def get_documents_by_type(self, data_type: str) -> List[Dict[str, Any]]:
        """
        Retrieve all documents of a specific data type.
        
        Args:
            data_type: Data type to filter by (e.g., 'rne_law', 'business_fiscal', 'fiscal_info').
            
        Returns:
            List of documents of the specified type.
        """
        if self.processed_data is None:
            self.process_data()
            
        return [doc for doc in self.processed_data if doc.get("data_type") == data_type]
    
    def get_documents_by_code(self, code: str) -> List[Dict[str, Any]]:
        """
        Retrieve all documents matching a given RNE code.
        
        Args:
            code: RNE code to match.
            
        Returns:
            List of matching document dictionaries.
        """
        if self.processed_data is None:
            self.process_data()
            
        return [doc for doc in self.processed_data if doc.get("code") == code]
    
    def get_documents_by_language(self, language: str) -> List[Dict[str, Any]]:
        """
        Retrieve all documents in a given language.
        
        Args:
            language: Language code ('fr' or 'ar').
            
        Returns:
            List of documents in the specified language.
        """
        if self.processed_data is None:
            self.process_data()
            
        return [doc for doc in self.processed_data if doc.get("language") == language]
    
    def extract_text_for_indexing(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract text content for indexing and keep reference to original documents.
        
        Returns:
            Tuple containing (list of text contents, list of corresponding documents)
        """
        if self.processed_data is None:
            self.process_data()
            
        texts = []
        docs = []
        
        for doc in self.processed_data:
            try:
                # Combine all relevant fields for rich text representation
                text_parts = []
                
                # Add basic identifying information
                if doc.get('code'):
                    text_parts.append(doc['code'])
                if doc.get('id'):
                    text_parts.append(doc['id'])
                
                # Add RNE-specific fields if they exist
                for field in ['type_entreprise', 'genre_entreprise', 'procedure']:
                    if doc.get(field):
                        text_parts.append(doc[field])
                
                # Add main content
                if doc.get('content'):
                    text_parts.append(doc['content'])
                
                # Filter out empty parts
                text_parts = [part.strip() for part in text_parts if part and str(part).strip()]
                
                # Join all parts with spaces
                text = ' '.join(text_parts)
                
                # Only add documents with non-empty text
                if text.strip():
                    texts.append(text)
                    docs.append(doc)
                else:
                    logging.warning(f"Empty text for document {doc.get('id', 'unknown')}. Skipping.")
                    
            except Exception as e:
                logging.error(f"Error processing document {doc.get('id', 'unknown')}: {str(e)}")
                # Continue with other documents
                continue
                
        if not texts:
            logging.warning("No text extracted for indexing. Check your data.")
            
        logging.info(f"Extracted {len(texts)} text documents for indexing")
        return texts, docs
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.
        
        Returns:
            Dictionary containing data statistics.
        """
        if self.processed_data is None:
            self.process_data()
            
        # Basic stats
        stats = {
            "total_documents": len(self.processed_data),
            "files_loaded": list(self.raw_data.keys()),
            "documents_by_source": {},
            "documents_by_type": {},
            "documents_by_language": {}
        }
        
        # Count by source file
        for filename in self.raw_data.keys():
            stats["documents_by_source"][filename] = len(self.get_documents_by_source(filename))
        
        # Count by data type
        data_types = set(doc.get("data_type", "unknown") for doc in self.processed_data)
        for data_type in data_types:
            stats["documents_by_type"][data_type] = len(self.get_documents_by_type(data_type))
        
        # Count by language
        languages = set(doc.get("language", "unknown") for doc in self.processed_data)
        for language in languages:
            stats["documents_by_language"][language] = len(self.get_documents_by_language(language))
        
        # RNE-specific stats
        rne_docs = self.get_documents_by_type("rne_law")
        if rne_docs:
            stats.update({
                "rne_documents": len(rne_docs),
                "unique_rne_codes": len(set(doc.get("code") for doc in rne_docs if doc.get("code"))),
                "rne_document_types": list(set(doc.get("type_entreprise", "") for doc in rne_docs if doc.get("type_entreprise")))
            })
        
        return stats


# Usage example:
if __name__ == "__main__":
    # Configuration - specify exact file paths
    DATA_FILES = [
        "./s/external_data.json",
        "./s/rne_laws.json",
    ]
    
    EXPECTED_DATA_FILES = {
        "external_data.json": "Business and fiscal knowledge",
        "rne_laws.json": "RNE legal procedures", 
        "fiscal_knowledge.json": "Additional fiscal information"
    }
    
    # Initialize with specific file paths
    loader = RNEDataLoader(data_paths=DATA_FILES, expected_files=EXPECTED_DATA_FILES)
    
    # Alternative: Load just the two specific files you mentioned
    # loader = RNEDataLoader(data_paths=["data/external_data.json", "data/rne_laws.json"])
    
    try:
        # Load all data
        raw_data = loader.load_data()
        print(f"Loaded data from {len(raw_data)} files")
        
        # Process data
        processed_docs = loader.process_data()
        print(f"Processed {len(processed_docs)} total documents")
        
        # Get statistics
        stats = loader.get_stats()
        print("Data Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Extract text for indexing
        texts, docs = loader.extract_text_for_indexing()
        print(f"Extracted {len(texts)} documents ready for indexing")
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")