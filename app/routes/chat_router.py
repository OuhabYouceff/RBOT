"""
Chat API router - FastAPI version.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from app.models.schemas import ChatRequest, ChatResponse, FollowUpRequest
from app.services.Chat_servicer import ChatServicer
from app.utils.response_formatter import ResponseFormatter

# Configure logging
logger = logging.getLogger(__name__)

# Create FastAPI router
chat_router = APIRouter(prefix="/api", tags=["chat-legacy"])

# Global chat service instance (will be injected)
chat_service: ChatServicer = None

def init_chat_router(service: ChatServicer):
    """Initialize the chat router with dependencies."""
    global chat_service
    chat_service = service
    logger.info("Chat router initialized")

def get_chat_service() -> ChatServicer:
    """Dependency to get chat service."""
    if chat_service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    return chat_service

@chat_router.post('/chatt', response_model=Dict[str, Any])
async def chat_legacy(request: ChatRequest, service: ChatServicer = Depends(get_chat_service)):
    """
    API endpoint for chatting with the RNE chatbot (legacy endpoint).
    
    Expects JSON with:
    - query: User query text
    - language: (Optional) Preferred language ('fr' or 'ar')
    - is_follow_up: (Optional) Boolean indicating if this is a follow-up
    - selected_option: (Optional) Selected option from clarification
    
    Returns:
    - JSON response with generated answer and metadata
    """
    try:
        # Log request
        logger.info(f"Processing chat request: {request.query[:100]}... (Language: {request.language})")
        
        # Process the chat request
        response = service.process_chat(request)
        
        logger.info("Chat request processed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        
        error_response = ResponseFormatter.format_error_response(
            f"Une erreur s'est produite: {str(e)}" if request.language == 'fr' else f"حدث خطأ: {str(e)}",
            request.language or 'fr'
        )
        raise HTTPException(status_code=500, detail=error_response)