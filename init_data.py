"""
Script to initialize the vector store and indices for the RNE chatbot.
This should be run once before starting the application.
"""

import os
import sys
import traceback
import json
import logging

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.config import DATA_PATH, FAISS_INDEX_PATH, BM25_DATA_PATH
from app.services.data_loader import RNEDataLoader
from app.services.faiss_retriever import FAISSRetriever
from app.services.bm25_retriever import BM25Retriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_json_file(file_path):
    """Verify that the JSON file exists and is valid."""
    if not os.path.exists(file_path):
        logger.error(f"File not found at {file_path}")
        return False
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            logger.info(f"JSON file is valid and contains {len(data)} items")
            # Check the first item for expected structure
            if data:
                first_item = data[0]
                expected_keys = ['code', 'type_entreprise', 'procedure']
                missing_keys = [key for key in expected_keys if key not in first_item]
                if missing_keys:
                    logger.warning(f"First item is missing expected keys: {missing_keys}")
                else:
                    logger.info("First item has expected structure")
                    
        elif isinstance(data, dict):
            logger.info("JSON file is valid and contains a single item")
            expected_keys = ['code', 'type_entreprise', 'procedure']
            missing_keys = [key for key in expected_keys if key not in data]
            if missing_keys:
                logger.warning(f"Item is missing expected keys: {missing_keys}")
            else:
                logger.info("Item has expected structure")
        else:
            logger.warning(f"JSON file has unexpected format. Expected a list or dictionary.")
            
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON file at {file_path}. Details: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error checking JSON file: {str(e)}")
        return False

def initialize_indices():
    """Initialize FAISS and BM25 indices."""
    logger.info("Initializing indices for the RNE chatbot...")
    
    # First verify the JSON file
    logger.info(f"Verifying JSON file at {DATA_PATH}...")
    if not verify_json_file(DATA_PATH):
        logger.error("Cannot proceed with initialization due to JSON file issues.")
        return False
    
    try:
        # Initialize data loader
        data_loader = RNEDataLoader(DATA_PATH)
        
        # Load and process data
        logger.info(f"Loading data from {DATA_PATH}...")
        raw_data = data_loader.load_data()
        
        if not raw_data:
            logger.error("No data loaded from file")
            return False
        
        logger.info("Processing data...")
        processed_data = data_loader.process_data()
        logger.info(f"Processed {len(processed_data)} documents")
        
        if not processed_data:
            logger.error("No documents were processed")
            return False
        
        # Extract text for indexing
        logger.info("Extracting text for indexing...")
        texts, docs = data_loader.extract_text_for_indexing()
        logger.info(f"Extracted {len(texts)} text entries for indexing")
        
        if not texts:
            logger.error("No text extracted for indexing")
            return False
        
        # Initialize retrievers
        faiss_retriever = FAISSRetriever(FAISS_INDEX_PATH)
        bm25_retriever = BM25Retriever(BM25_DATA_PATH)
        
        # Build indices
        logger.info("Building FAISS index...")
        faiss_retriever.build_index(texts, docs)
        
        logger.info("Building BM25 index...")
        bm25_retriever.build_index(texts, docs)
        
        logger.info("Indices built and saved successfully!")
        
        # Test retrieval
        test_queries = [
            "documents pour immatriculation sarl",
            "capital minimum société anonyme",
            "délai création entreprise"
        ]
        
        for test_query in test_queries:
            logger.info(f"\nTesting retrieval with query: {test_query}")
            
            # Test FAISS
            try:
                faiss_results = faiss_retriever.search(test_query, top_k=2)
                logger.info(f"FAISS found {len(faiss_results)} results:")
                for i, result in enumerate(faiss_results):
                    logger.info(f"  Result {i+1}: {result['document']['code']} (Score: {result['score']:.4f})")
            except Exception as e:
                logger.error(f"Error testing FAISS: {e}")
                
            # Test BM25
            try:
                bm25_results = bm25_retriever.search(test_query, top_k=2)
                logger.info(f"BM25 found {len(bm25_results)} results:")
                for i, result in enumerate(bm25_results):
                    logger.info(f"  Result {i+1}: {result['document']['code']} (Score: {result['score']:.4f})")
            except Exception as e:
                logger.error(f"Error testing BM25: {e}")
        
        # Print statistics
        logger.info("\nData Statistics:")
        stats = data_loader.get_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
            
        logger.info("\nInitialization complete!")
        return True
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        logger.error("Traceback:")
        traceback.print_exc()
        return False

def ensure_directories_exist():
    """Ensure all necessary directories exist."""
    directories = [
        os.path.dirname(DATA_PATH),
        os.path.dirname(FAISS_INDEX_PATH),
        os.path.dirname(BM25_DATA_PATH)
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)

def main():
    """Main initialization function."""
    logger.info("Starting RNE Chatbot initialization...")
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Initialize indices
    success = initialize_indices()
    
    if success:
        logger.info("✅ Initialization completed successfully!")
        logger.info("You can now start the application with: python -m app.main")
    else:
        logger.error("❌ Initialization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()