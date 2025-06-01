"""
Models package for data structures and schemas.
"""

from .schemas import (
    ChatRequest,
    ChatResponse,
    FollowUpRequest,
    FollowUpResponse,
    DocumentReference,
    ErrorResponse,
    HealthResponse
)
from .data_models import (
    RNEDocument,
    RetrievalResult,
    ProcessedDocument
)

__all__ = [
    'ChatRequest',
    'ChatResponse', 
    'FollowUpRequest',
    'FollowUpResponse',
    'DocumentReference',
    'ErrorResponse',
    'HealthResponse',
    'RNEDocument',
    'RetrievalResult',
    'ProcessedDocument'
]