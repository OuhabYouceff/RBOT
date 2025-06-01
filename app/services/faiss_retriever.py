"""
Vector-based retrieval system using FAISS.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logging.warning("FAISS not available. Vector retrieval will not work.")
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("sentence-transformers not available. Vector retrieval will not work.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from app.utils.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION

class FAISSRetriever:
    """
    Vector-based retrieval system using FAISS for semantic search.
    """
    
    def __init__(self, index_path=None, embedding_model=EMBEDDING_MODEL):
        """
        Initialize the FAISS retriever.
        
        Args:
            index_path: Path to save/load the FAISS index.
            embedding_model: Name of the sentence transformer model to use.
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS package is required for FAISSRetriever. Install with: pip install faiss-cpu")
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers package is required for FAISSRetriever. Install with: pip install sentence-transformers")
            
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.index = None
        self.documents = []
        
        try:
            self.encoder = SentenceTransformer(embedding_model)
            logging.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logging.error(f"Error loading embedding model {embedding_model}: {e}")
            raise
        
    def build_index(self, texts: List[str], documents: List[Dict[str, Any]]) -> None:
        """
        Build a FAISS index from the provided texts and documents.
        
        Args:
            texts: List of text strings to index.
            documents: List of document dictionaries corresponding to the texts.
        """
        if not texts or not documents:
            logging.warning("No texts or documents provided for indexing")
            return
            
        if len(texts) != len(documents):
            raise ValueError("Number of texts and documents must match")
            
        try:
            # Store the documents for later retrieval
            self.documents = documents
            
            # Filter out empty texts
            filtered_texts = []
            filtered_docs = []
            for i, text in enumerate(texts):
                if text and text.strip():
                    filtered_texts.append(text)
                    filtered_docs.append(documents[i])
            
            if not filtered_texts:
                logging.warning("No valid texts found for indexing")
                return
                
            self.documents = filtered_docs
            
            # Create embeddings for all texts
            logging.info(f"Creating embeddings for {len(filtered_texts)} documents...")
            embeddings = self._create_embeddings(filtered_texts)
            
            if embeddings is None or len(embeddings) == 0:
                logging.error("Failed to create embeddings")
                return
            
            # Create and configure the FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity when normalized)
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add vectors to the index
            self.index.add(embeddings)
            logging.info(f"Added {len(embeddings)} vectors to FAISS index")
            
            # Save the index if a path is provided
            if self.index_path:
                self._save_index()
                
        except Exception as e:
            logging.error(f"Error building FAISS index: {e}")
            raise
    
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to encode.
            
        Returns:
            NumPy array of embeddings.
        """
        try:
            embeddings = self.encoder.encode(
                texts, 
                convert_to_numpy=True, 
                show_progress_bar=True,
                batch_size=32  # Process in smaller batches to avoid memory issues
            )
            return embeddings
        except Exception as e:
            logging.error(f"Error creating embeddings: {e}")
            return None
    
    def _save_index(self) -> None:
        """Save the FAISS index and documents to disk."""
        try:
            if not self.index_path:
                return
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save the FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save the documents in a separate file
            documents_path = self.index_path.replace('.bin', '_docs.pkl')
            with open(documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
                
            logging.info(f"Saved FAISS index to {self.index_path}")
            
        except Exception as e:
            logging.error(f"Error saving FAISS index: {e}")
    
    def load_index(self) -> bool:
        """
        Load the FAISS index and documents from disk.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.index_path or not os.path.exists(self.index_path):
            logging.warning(f"FAISS index file not found at {self.index_path}")
            return False
            
        try:
            # Load the FAISS index
            self.index = faiss.read_index(self.index_path)
            
            # Load the documents
            documents_path = self.index_path.replace('.bin', '_docs.pkl')
            if os.path.exists(documents_path):
                with open(documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
            else:
                logging.warning(f"Documents file not found at {documents_path}")
                self.documents = []
                
            logging.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logging.error(f"Error loading FAISS index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3, language: str = None) -> List[Dict[str, Any]]:
        """
        Search the index for documents similar to the query.
        
        Args:
            query: Query string.
            top_k: Number of top results to return.
            language: Optional language filter ('fr' or 'ar').
            
        Returns:
            List of dictionaries with document info and scores.
        """
        if not self.index:
            logging.warning("FAISS index not built or loaded")
            return []
            
        if not query or not query.strip():
            return []
            
        try:
            # Create query embedding
            query_embedding = self._create_embeddings([query])
            
            if query_embedding is None or len(query_embedding) == 0:
                logging.error("Failed to create query embedding")
                return []
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search the index - get more results for language filtering
            search_k = top_k * 3 if language else top_k
            scores, indices = self.index.search(query_embedding, min(search_k, self.index.ntotal))
            
            # Extract results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < 0 or idx >= len(self.documents):
                    continue
                    
                document = self.documents[idx]
                
                # Filter by language if specified
                if language and document.get('language') != language:
                    continue
                    
                results.append({
                    'document': document,
                    'score': float(score),  # Convert from numpy float to Python float
                    'rank': len(results) + 1
                })
                
                # Stop once we have enough results after filtering
                if len(results) >= top_k:
                    break
                    
            return results
            
        except Exception as e:
            logging.error(f"Error in FAISS search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the FAISS index.
        
        Returns:
            Dictionary containing index statistics.
        """
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_documents": len(self.documents),
            "embedding_model": self.embedding_model,
            "index_built": self.index is not None
        }