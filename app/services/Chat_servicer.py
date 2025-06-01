"""
Chat service for handling chat logic.
"""

import logging
from typing import Dict, Any, List

from app.models.schemas import ChatRequest
from app.models.data_models import QueryContext
from app.services.openai_client import OpenAIClient, DirectResponse, FollowUpResponse
from app.services.hybrid_retriever import HybridRetriever
from app.utils.language_detector import LanguageDetector
from app.utils.response_formatter import ResponseFormatter
from app.utils.prompt_templates import get_no_results_response
from app.utils.config import TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)

class ChatServicer:
    """Service for handling chat operations."""
    
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        openai_client: OpenAIClient,
        language_detector: LanguageDetector
    ):
        """
        Initialize the chat service.
        
        Args:
            hybrid_retriever: Hybrid retrieval system
            openai_client: OpenAI client
            language_detector: Language detector
        """
        self.hybrid_retriever = hybrid_retriever
        self.openai_client = openai_client
        self.language_detector = language_detector
        
        logger.info("Chat service initialized")
    
    def process_chat(self, chat_request: ChatRequest) -> Dict[str, Any]:
        """
        Process a chat request.
        
        Args:
            chat_request: Validated chat request
            
        Returns:
            Formatted response dictionary
        """
        try:
            # Create query context
            context = self._create_query_context(chat_request)
            
            # Detect language if not provided
            if not context.language:
                context.language = self.language_detector.detect_language(context.original_query)
            
            # Process based on whether it's a follow-up or regular query
            if context.is_follow_up:
                return self._process_follow_up(context)
            else:
                return self._process_regular_query(context)
                
        except Exception as e:
            logger.error(f"Error processing chat request: {e}")
            language = chat_request.language or 'fr'
            return ResponseFormatter.format_error_response(
                f"Erreur lors du traitement: {str(e)}" if language == 'fr' else f"خطأ في المعالجة: {str(e)}",
                language
            )
    
    def segment_questions(self, query: str) -> List[str]:
        """
        Segment a query into multiple questions.
        
        Args:
            query: User query
            
        Returns:
            List of segmented questions
        """
        try:
            return self.openai_client.segment_questions(query)
        except Exception as e:
            logger.error(f"Error segmenting questions: {e}")
            return [query]  # Return original query as fallback
    
    def _create_query_context(self, chat_request: ChatRequest) -> QueryContext:
        """Create query context from chat request."""
        processed_query = chat_request.query
        
        # Combine with selected option if it's a follow-up
        if chat_request.is_follow_up and chat_request.selected_option:
            processed_query = f"{chat_request.query} - {chat_request.selected_option}"
        
        return QueryContext(
            original_query=chat_request.query,
            processed_query=processed_query,
            language=chat_request.language or '',
            is_follow_up=chat_request.is_follow_up or False,
            selected_option=chat_request.selected_option
        )
    
    def _process_follow_up(self, context: QueryContext) -> Dict[str, Any]:
        """Process a follow-up query."""
        try:
            # Retrieve relevant documents
            results = self.hybrid_retriever.search(
                context.processed_query, 
                TOP_K_RETRIEVAL, 
                context.language
            )
            
            # Generate response (force direct for follow-ups)
            response_obj = self.openai_client.generate_response(
                context.processed_query, 
                results, 
                context.language, 
                force_direct=True
            )
            
            # Extract answer
            if isinstance(response_obj, DirectResponse):
                answer = response_obj.response
            else:
                answer = str(response_obj)
            
            # Use default message if no results
            if not results:
                answer = get_no_results_response(context.language)
            
            # Format response
            return ResponseFormatter.format_response(
                answer, 
                context.original_query, 
                results, 
                context.language
            )
            
        except Exception as e:
            logger.error(f"Error processing follow-up: {e}")
            raise
    
    def _process_regular_query(self, context: QueryContext) -> Dict[str, Any]:
        """Process a regular (non-follow-up) query."""
        try:
            # Check if query contains multiple questions
            questions = self.openai_client.segment_questions(context.original_query)
            
            if len(questions) == 1:
                return self._process_single_question(context)
            else:
                return self._process_multiple_questions(questions, context)
                
        except Exception as e:
            logger.error(f"Error processing regular query: {e}")
            raise
    
    def _process_single_question(self, context: QueryContext) -> Dict[str, Any]:
        """Process a single question."""
        try:
            # Retrieve relevant documents
            results = self.hybrid_retriever.search(
                context.processed_query, 
                TOP_K_RETRIEVAL, 
                context.language
            )
            
            # Generate response
            response_obj = self.openai_client.generate_response(
                context.processed_query, 
                results, 
                context.language
            )
            
            # Handle different response types
            if isinstance(response_obj, FollowUpResponse):
                # Return clarification response
                return ResponseFormatter.format_clarification_response(
                    response_obj.main_response,
                    response_obj.follow_up_question,
                    response_obj.options,
                    context.language
                )
                
            elif isinstance(response_obj, DirectResponse):
                answer = response_obj.response
            else:
                answer = str(response_obj)
            
            # Use default message if no results
            if not results:
                answer = get_no_results_response(context.language)
            
            # Format response
            return ResponseFormatter.format_response(
                answer, 
                context.original_query, 
                results, 
                context.language
            )
            
        except Exception as e:
            logger.error(f"Error processing single question: {e}")
            raise
    
    def _process_multiple_questions(self, questions: List[str], context: QueryContext) -> Dict[str, Any]:
        """Process multiple questions."""
        try:
            responses = []
            
            for question in questions:
                # Retrieve relevant documents for each question
                results = self.hybrid_retriever.search(
                    question, 
                    TOP_K_RETRIEVAL, 
                    context.language
                )
                
                # Generate response (force direct for multi-question)
                response_obj = self.openai_client.generate_response(
                    question, 
                    results, 
                    context.language, 
                    force_direct=True
                )
                
                # Extract answer
                if isinstance(response_obj, DirectResponse):
                    answer = response_obj.response
                else:
                    answer = str(response_obj)
                
                # Use default message if no results
                if not results:
                    answer = get_no_results_response(context.language)
                
                # Format individual response
                formatted_response = ResponseFormatter.format_response(
                    answer, 
                    question, 
                    results, 
                    context.language
                )
                responses.append(formatted_response)
            
            # Combine responses
            return ResponseFormatter.format_multi_response(
                responses, 
                context.original_query, 
                context.language
            )
            
        except Exception as e:
            logger.error(f"Error processing multiple questions: {e}")
            raise