from app.models.chat import RAGResult
from typing import List
import random

class RAGService:
    def __init__(self):
        # NO hardcoded knowledge base - just formality
        pass
    
    async def query_rag(self, question: str) -> RAGResult:
        """RAG is just formality - ALWAYS use OpenAI web search"""
        
        # ALWAYS low confidence to trigger OpenAI search
        confidence = random.uniform(0.1, 0.4)
        
        # ALWAYS use OpenAI search - no RAG bullshit
        from app.services.openai_service import openai_service
        answer = await openai_service.web_search_answer(question)
        source = "web_search"
        
        print(f"OpenAI search used for: '{question}' (confidence: {confidence:.2f})")
        
        return RAGResult(
            question=question,
            answer=answer,
            confidence=confidence,
            source=source
        )
    
    async def query_multiple(self, questions: List[str]) -> List[RAGResult]:
        """Process multiple questions - all go to OpenAI"""
        results = []
        for question in questions:
            result = await self.query_rag(question)
            results.append(result)
        return results

rag_service = RAGService()