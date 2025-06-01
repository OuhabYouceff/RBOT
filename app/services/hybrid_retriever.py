"""
Hybrid retrieval system combining FAISS and BM25.
"""

from typing import List, Dict, Any
import logging

from app.utils.config import FAISS_WEIGHT, BM25_WEIGHT, TOP_K_RETRIEVAL
from app.services.faiss_retriever import FAISSRetriever
from app.services.bm25_retriever import BM25Retriever
from app.services.text_processor import TextProcessor

class HybridRetriever:
    """
    Hybrid retrieval system combining semantic search (FAISS) and keyword search (BM25).
    """
    
    def __init__(
        self, 
        faiss_retriever: FAISSRetriever = None, 
        bm25_retriever: BM25Retriever = None,
        faiss_weight: float = FAISS_WEIGHT,
        bm25_weight: float = BM25_WEIGHT
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            faiss_retriever: Initialized FAISS retriever.
            bm25_retriever: Initialized BM25 retriever.
            faiss_weight: Weight for FAISS results in hybrid scoring.
            bm25_weight: Weight for BM25 results in hybrid scoring.
        """
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight
        self.text_processor = TextProcessor()
        
        # Validate that at least one retriever is available
        if not self.faiss_retriever and not self.bm25_retriever:
            raise ValueError("At least one retriever (FAISS or BM25) must be provided")
            
        logging.info(f"Initialized HybridRetriever with FAISS weight: {faiss_weight}, BM25 weight: {bm25_weight}")
        
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL, language: str = None) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search using both FAISS and BM25.
        
        Args:
            query: Query string.
            top_k: Number of top results to return.
            language: Optional language filter ('fr' or 'ar'). If None, detect language from query.
            
        Returns:
            List of dictionaries with document info and combined scores.
        """
        if not query or not query.strip():
            return []
            
        try:
            # Detect language if not specified
            if not language:
                language = self.text_processor.detect_language(query)
                
            # Increase the number of results to retrieve from each system to ensure we have enough for reranking
            internal_top_k = max(top_k * 2, 10)
            
            # Get results from both retrieval systems
            faiss_results = []
            bm25_results = []
            
            if self.faiss_retriever:
                try:
                    faiss_results = self.faiss_retriever.search(query, internal_top_k, language)
                except Exception as e:
                    logging.error(f"Error in FAISS search: {e}")
                    
            if self.bm25_retriever:
                try:
                    bm25_results = self.bm25_retriever.search(query, internal_top_k, language)
                except Exception as e:
                    logging.error(f"Error in BM25 search: {e}")
            
            # If only one retriever is available, return its results
            if not faiss_results and bm25_results:
                return bm25_results[:top_k]
            elif faiss_results and not bm25_results:
                return faiss_results[:top_k]
            elif not faiss_results and not bm25_results:
                logging.warning(f"No results found for query: {query}")
                return []
            
            # Combine and rerank results
            combined_results = self._combine_results(faiss_results, bm25_results, top_k)
            
            return combined_results
            
        except Exception as e:
            logging.error(f"Error in hybrid search: {e}")
            return []
    
    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize scores to be between 0 and 1.
        
        Args:
            results: List of result dictionaries with scores.
            
        Returns:
            List of result dictionaries with normalized scores.
        """
        if not results:
            return []
            
        try:
            # Extract scores
            scores = [result['score'] for result in results]
            
            # Find min and max scores
            min_score = min(scores)
            max_score = max(scores)
            
            # Avoid division by zero
            if max_score == min_score:
                # If all scores are the same, set them to 1.0
                for result in results:
                    result['score'] = 1.0
                return results
                
            # Normalize scores to [0, 1]
            for result in results:
                result['score'] = (result['score'] - min_score) / (max_score - min_score)
                
            return results
            
        except Exception as e:
            logging.error(f"Error normalizing scores: {e}")
            return results
    
    def _combine_results(
        self, 
        faiss_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine and rerank results from both retrieval systems.
        
        Args:
            faiss_results: Results from FAISS retriever.
            bm25_results: Results from BM25 retriever.
            top_k: Number of top results to return.
            
        Returns:
            List of combined and reranked results.
        """
        try:
            # Normalize scores within each system
            faiss_results = self._normalize_scores(faiss_results.copy())
            bm25_results = self._normalize_scores(bm25_results.copy())
            
            # Create a dictionary to store combined scores
            doc_scores = {}
            
            # Add FAISS scores
            for result in faiss_results:
                doc_id = result['document']['id']
                doc_scores[doc_id] = {
                    'document': result['document'],
                    'faiss_score': result['score'] * self.faiss_weight,
                    'bm25_score': 0,
                    'combined_score': 0
                }
                
            # Add BM25 scores
            for result in bm25_results:
                doc_id = result['document']['id']
                if doc_id in doc_scores:
                    doc_scores[doc_id]['bm25_score'] = result['score'] * self.bm25_weight
                else:
                    doc_scores[doc_id] = {
                        'document': result['document'],
                        'faiss_score': 0,
                        'bm25_score': result['score'] * self.bm25_weight,
                        'combined_score': 0
                    }
                    
            # Calculate combined scores
            for doc_id, data in doc_scores.items():
                data['combined_score'] = data['faiss_score'] + data['bm25_score']
                
            # Sort by combined score
            sorted_results = sorted(
                doc_scores.values(), 
                key=lambda x: x['combined_score'], 
                reverse=True
            )
            
            # Format the final results
            final_results = []
            for i, result in enumerate(sorted_results[:top_k]):
                final_results.append({
                    'document': result['document'],
                    'score': result['combined_score'],
                    'faiss_score': result['faiss_score'],
                    'bm25_score': result['bm25_score'],
                    'rank': i + 1
                })
                
            logging.info(f"Combined {len(faiss_results)} FAISS and {len(bm25_results)} BM25 results into {len(final_results)} hybrid results")
            return final_results
            
        except Exception as e:
            logging.error(f"Error combining results: {e}")
            # Fallback: return FAISS results if available, otherwise BM25
            if faiss_results:
                return faiss_results[:top_k]
            else:
                return bm25_results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the hybrid retriever.
        
        Returns:
            Dictionary containing retriever statistics.
        """
        stats = {
            "faiss_available": self.faiss_retriever is not None,
            "bm25_available": self.bm25_retriever is not None,
            "faiss_weight": self.faiss_weight,
            "bm25_weight": self.bm25_weight
        }
        
        if self.faiss_retriever:
            stats["faiss_stats"] = self.faiss_retriever.get_stats()
            
        if self.bm25_retriever:
            stats["bm25_stats"] = self.bm25_retriever.get_stats()
            
        return stats