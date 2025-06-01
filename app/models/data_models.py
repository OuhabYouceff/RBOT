"""
Data models for internal use.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

class ResponseType(Enum):
    """Enum for response types."""
    DIRECT_ANSWER = "direct_answer"
    CLARIFICATION_NEEDED = "clarification_needed"
    NO_RESULTS = "no_results"
    ERROR = "error"

class Language(Enum):
    """Enum for supported languages."""
    FRENCH = "fr"
    ARABIC = "ar"

@dataclass
class RNEDocument:
    """Data model for RNE document."""
    id: str
    code: str
    language: str
    type_entreprise: str
    genre_entreprise: str
    procedure: str
    redevance_demandee: str
    delais: str
    pdf_link: str
    content: str
    raw_content: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'code': self.code,
            'language': self.language,
            'type_entreprise': self.type_entreprise,
            'genre_entreprise': self.genre_entreprise,
            'procedure': self.procedure,
            'redevance_demandee': self.redevance_demandee,
            'delais': self.delais,
            'pdf_link': self.pdf_link,
            'content': self.content,
            'raw_content': self.raw_content
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RNEDocument':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            code=data['code'],
            language=data['language'],
            type_entreprise=data.get('type_entreprise', ''),
            genre_entreprise=data.get('genre_entreprise', ''),
            procedure=data.get('procedure', ''),
            redevance_demandee=data.get('redevance_demandee', ''),
            delais=data.get('delais', ''),
            pdf_link=data.get('pdf_link', ''),
            content=data.get('content', ''),
            raw_content=data.get('raw_content', {})
        )

@dataclass
class RetrievalResult:
    """Data model for retrieval result."""
    document: RNEDocument
    score: float
    rank: int
    faiss_score: Optional[float] = None
    bm25_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'document': self.document.to_dict(),
            'score': self.score,
            'rank': self.rank
        }
        
        if self.faiss_score is not None:
            result['faiss_score'] = self.faiss_score
        if self.bm25_score is not None:
            result['bm25_score'] = self.bm25_score
            
        return result

@dataclass
class ProcessedDocument:
    """Data model for processed document."""
    text: str
    document: RNEDocument
    tokens: List[str]
    language: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'document': self.document.to_dict(),
            'tokens': self.tokens,
            'language': self.language
        }

@dataclass
class QueryContext:
    """Data model for query context."""
    original_query: str
    processed_query: str
    language: str
    is_follow_up: bool
    selected_option: Optional[str] = None
    detected_intent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'original_query': self.original_query,
            'processed_query': self.processed_query,
            'language': self.language,
            'is_follow_up': self.is_follow_up,
            'selected_option': self.selected_option,
            'detected_intent': self.detected_intent
        }