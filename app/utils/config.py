"""
Configuration settings for the RNE Chatbot application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='allow'  # This allows extra fields from .env
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = DATA_DIR  # The data loader will scan this directory for JSON files

# Expected data files (for reference and validation)
EXPECTED_DATA_FILES = {
    "external_data.json": "Business and fiscal knowledge",
    "rne_laws.json": "RNE legal procedures", 
    "fiscal_knowledge.json": "Additional fiscal information"
}
# Validate API key
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables")

# Retrieval settings
FAISS_WEIGHT = float(os.getenv("FAISS_WEIGHT", "0.5"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))

# Vector embedding settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# Language settings
SUPPORTED_LANGUAGES = ["fr", "ar"]
DEFAULT_LANGUAGE = "fr"

# Flask settings
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5001"))

# Data paths
DATA_DIR = os.getenv("DATA_DIR", "data")
DATA_PATH = os.getenv("DATA_PATH", os.path.join(DATA_DIR, "rne_laws.json"))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.join(DATA_DIR, "faiss_index.bin"))
BM25_DATA_PATH = os.getenv("BM25_DATA_PATH", os.path.join(DATA_DIR, "bm25_data.pkl"))

# Prompt settings
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "4096"))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are an expert legal assistant specializing in Tunisian RNE laws.""")
