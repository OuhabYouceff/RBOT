from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ConversationMessage(BaseModel):
    type: str  # "user" or "bot"
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "auto"
    conversation_history: Optional[List[ConversationMessage]] = []

class FollowUpQuestion(BaseModel):
    question: str
    options: List[str]

class RNEFormData(BaseModel):
    code: str
    title: str
    subtitle: str
    url: str

class ChatResponse(BaseModel):
    answer: str
    follow_up: Optional[FollowUpQuestion] = None
    suggestions: List[str] = []
    forms: List[RNEFormData] = []
    status: str = "success"

class ErrorResponse(BaseModel):
    error: str
    status: str = "error"

# Internal pipeline models
class SegmentedQuestions(BaseModel):
    questions: List[str]
    original_query: str

class RAGResult(BaseModel):
    question: str
    answer: str
    confidence: float
    source: str  # "rag" or "web_search"