from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routes import chat
from app.services.Chat_servicer import ChatServicer

from app.utils.config import DEBUG, HOST, PORT

from app.services import (
    RNEDataLoader, TextProcessor, FAISSRetriever, 
    BM25Retriever, HybridRetriever, OpenAIClient
)
from app.utils import LanguageDetector
from app.utils.config import DATA_PATH, FAISS_INDEX_PATH, BM25_DATA_PATH
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def initialize_components():
    """Initialize all components."""
    logger.info("Initializing components...")
    
    # Initialize basic components
    text_processor = TextProcessor()
    language_detector = LanguageDetector()
    data_loader = RNEDataLoader(DATA_PATH)
    
    # Initialize retrievers
    faiss_retriever = FAISSRetriever(FAISS_INDEX_PATH)
    bm25_retriever = BM25Retriever(BM25_DATA_PATH)
    
    # Load or build indices
    faiss_loaded = faiss_retriever.load_index()
    bm25_loaded = bm25_retriever.load_index()
    
    if not (faiss_loaded and bm25_loaded):
        logger.warning("Building indices...")
        data_loader.load_data()
        data_loader.process_data()
        texts, docs = data_loader.extract_text_for_indexing()
        
        if not faiss_loaded:
            faiss_retriever.build_index(texts, docs)
        if not bm25_loaded:
            bm25_retriever.build_index(texts, docs)
    
    # Initialize hybrid retriever and OpenAI client
    hybrid_retriever = HybridRetriever(faiss_retriever, bm25_retriever)
    openai_client = OpenAIClient()
    
    components = {
        'data_loader': data_loader,
        'text_processor': text_processor,
        'language_detector': language_detector,
        'faiss_retriever': faiss_retriever,
        'bm25_retriever': bm25_retriever,
        'hybrid_retriever': hybrid_retriever,
        'openai_client': openai_client
    }
    
    logger.info("All components initialized successfully!")
    return components

# Initialize components
components = initialize_components()

# Create chat service
chat_service = ChatServicer(
    hybrid_retriever=components['hybrid_retriever'],
    openai_client=components['openai_client'], 
    language_detector=components['language_detector']
)

# Include routers
app.include_router(chat.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.project_name}",
        "version": settings.version,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.version}