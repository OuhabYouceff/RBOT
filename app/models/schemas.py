"""
Pydantic schemas for request/response validation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator

class ChatRequest(BaseModel):
    """Schema for chat request."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    language: Optional[str] = Field(None, description="Preferred language ('fr' or 'ar')")
    is_follow_up: Optional[bool] = Field(False, description="Whether this is a follow-up question")
    selected_option: Optional[str] = Field(None, description="Selected option from clarification")
    
    @validator('query')
    def query_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    @validator('language')
    def language_must_be_supported(cls, v):
        if v and v not in ['fr', 'ar']:
            raise ValueError('Language must be "fr" or "ar"')
        return v

class DocumentReference(BaseModel):
    """Schema for document reference in response."""
    code: str = Field(..., description="RNE code")
    procedure: str = Field("", description="Procedure description")
    type_entreprise: str = Field("", description="Enterprise type")
    score: float = Field(..., ge=0, description="Relevance score")
    pdf_link: str = Field("", description="PDF link")
    language: str = Field(..., description="Document language")

class ChatResponse(BaseModel):
    """Schema for regular chat response."""
    response: str = Field(..., description="Generated response")
    language: str = Field(..., description="Response language")
    text_direction: str = Field(..., description="Text direction (ltr/rtl)")
    references: List[DocumentReference] = Field(default=[], description="Document references")
    query: str = Field(..., description="Original query")
    document_count: int = Field(..., ge=0, description="Number of documents used")
    referenced_codes: List[str] = Field(default=[], description="RNE codes mentioned in response")

class FollowUpResponse(BaseModel):
    """Schema for clarification/follow-up response."""
    type: str = Field("clarification_needed", description="Response type")
    response: str = Field(..., description="Main response explaining need for clarification")
    follow_up_question: str = Field(..., description="Question asking for clarification")
    options: List[str] = Field(..., description="Options for user to choose from")
    language: str = Field(..., description="Response language")
    text_direction: str = Field(..., description="Text direction (ltr/rtl)")
    awaiting_clarification: bool = Field(True, description="Whether waiting for clarification")

class MultiQuestionResponse(BaseModel):
    """Schema for multi-question response."""
    response: str = Field(..., description="Combined response")
    language: str = Field(..., description="Response language")
    text_direction: str = Field(..., description="Text direction (ltr/rtl)")
    references: List[DocumentReference] = Field(default=[], description="All document references")
    query: str = Field(..., description="Original query")
    question_count: int = Field(..., ge=1, description="Number of questions processed")
    document_count: int = Field(..., ge=0, description="Total documents used")
    referenced_codes: List[str] = Field(default=[], description="All RNE codes mentioned")

class ErrorResponse(BaseModel):
    """Schema for error response."""
    type: str = Field("error", description="Response type")
    response: str = Field(..., description="Error message")
    language: str = Field(..., description="Response language")
    text_direction: str = Field(..., description="Text direction (ltr/rtl)")
    error: bool = Field(True, description="Error flag")

class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str = Field(..., description="Health status")
    components: Dict[str, bool] = Field(..., description="Component status")
    message: str = Field(..., description="Status message")

class FollowUpRequest(BaseModel):
    """Schema for follow-up request after clarification."""
    original_query: str = Field(..., min_length=1, description="Original user query")
    selected_option: str = Field(..., min_length=1, description="Selected option")
    language: Optional[str] = Field(None, description="Preferred language")
    
    @validator('original_query', 'selected_option')
    def must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()