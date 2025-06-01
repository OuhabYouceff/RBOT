"""
Script to initialize the vector store and indices for the RNE chatbot.
This should be run once before starting the application.
"""

import os
import sys
import traceback
import json
import logging

# Add the current directory to Python path so we can import app modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now we can import app modules
try:
    from app.utils.config import DATA_PATH, FAISS_INDEX_PATH, BM25_DATA_PATH
    from app.services.data_loader import RNEDataLoader
    from app.services.faiss_retriever import FAISSRetriever
    from app.services.bm25_retriever import BM25Retriever
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory")
    print("Current directory:", current_dir)
    print("Directory contents:", os.listdir(current_dir))
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_project_structure():
    """Verify that the project structure is correct."""
    required_paths = [
        'app',
        'app/utils',
        'app/services',
        'app/models',
        'app/routers'
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        logger.error("Missing required project structure:")
        for path in missing_paths:
            logger.error(f"  - {path}")
        return False
    
    logger.info("Project structure verified")
    return True

def verify_json_file(file_path):
    """Verify that the JSON file exists and is valid."""
    if not os.path.exists(file_path):
        logger.error(f"File not found at {file_path}")
        logger.info("Please make sure your RNE laws data file is placed at this location")
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
                    logger.info("Expected structure should include: code, type_entreprise, procedure")
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
    
    # Verify project structure first
    if not verify_project_structure():
        logger.error("Cannot proceed with initialization due to project structure issues.")
        return False
    
    # Check if data file exists
    logger.info(f"Verifying JSON file at {DATA_PATH}...")
    if not verify_json_file(DATA_PATH):
        logger.error("Cannot proceed with initialization due to JSON file issues.")
        logger.info(f"Please place your RNE laws data file at: {DATA_PATH}")
        return False
    
    try:
        # Initialize data loader
        logger.info("Creating data loader...")
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
        logger.info("Initializing FAISS retriever...")
        faiss_retriever = FAISSRetriever(FAISS_INDEX_PATH)
        
        logger.info("Initializing BM25 retriever...")
        bm25_retriever = BM25Retriever(BM25_DATA_PATH)
        
        # Build indices
        logger.info("Building FAISS index... (this may take a few minutes)")
        faiss_retriever.build_index(texts, docs)
        logger.info("FAISS index built successfully!")
        
        logger.info("Building BM25 index...")
        bm25_retriever.build_index(texts, docs)
        logger.info("BM25 index built successfully!")
        
        logger.info("All indices built and saved successfully!")
        
        # Test retrieval
        test_queries = [
            "documents pour immatriculation sarl",
            "capital minimum société anonyme",
            "délai création entreprise"
        ]
        
        logger.info("Testing retrieval systems...")
        
        for test_query in test_queries:
            logger.info(f"\nTesting with query: '{test_query}'")
            
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
        logger.info("\n" + "="*50)
        logger.info("DATA STATISTICS")
        logger.info("="*50)
        stats = data_loader.get_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
            
        logger.info("\n" + "="*50)
        logger.info("✅ INITIALIZATION COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        logger.info("You can now start the application with:")
        logger.info("  python run.py")
        logger.info("or")
        logger.info("  python -m app.main")
        return True
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        logger.error("Full traceback:")
        traceback.print_exc()
        return False

def ensure_directories_exist():
    """Ensure all necessary directories exist."""
    directories = [
        os.path.dirname(DATA_PATH),
        os.path.dirname(FAISS_INDEX_PATH), 
        os.path.dirname(BM25_DATA_PATH),
        'logs'  # For logging
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'flask',
        'openai', 
        'numpy',
        'faiss',
        'sentence_transformers',
        'rank_bm25',
        'langdetect',
        'nltk'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'faiss':
                __import__('faiss')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error("Missing required packages:")
        for package in missing_packages:
            if package == 'faiss':
                logger.error(f"  - {package} (install with: pip install faiss-cpu)")
            else:
                logger.error(f"  - {package}")
        logger.error("\nInstall missing packages with:")
        logger.error("pip install -r requirements.txt")
        return False
    
    logger.info("All required dependencies are installed")
    return True

def main():
    """Main initialization function."""
    print("🚀 Starting RNE Chatbot Initialization...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")
    print()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed!")
        sys.exit(1)
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Initialize indices
    success = initialize_indices()
    
    if success:
        print("\n" + "🎉" * 20)
        print("✅ INITIALIZATION COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        print("\nNext steps:")
        print("1. Make sure your .env file has OPENAI_API_KEY set")
        print("2. Run the application: python run.py")
        print("3. Test the API: curl http://localhost:5000/api/health")
    else:
        print("\n" + "❌" * 20) 
        print("❌ INITIALIZATION FAILED!")
        print("❌" * 20)
        print("\nPlease check the errors above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()