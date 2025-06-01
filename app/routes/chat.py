from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse, ErrorResponse
from app.services.chat_pipeline_service import chat_pipeline_service
from app.services.forms_service import forms_service
from typing import Union

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=Union[ChatResponse, ErrorResponse])
async def chat_endpoint(request: ChatRequest):
    """
    Process chat query through the complete pipeline:
    1. Check if needs additional info
    2. Segment into questions 
    3. RAG + Web search
    4. Format final answer + suggestions
    5. Include relevant RNE forms when applicable
    """
    try:
        print(f"🔥 Received request: {request.message}")
        
        result = await chat_pipeline_service.process_chat(request)
        
        print(f"✅ Pipeline result type: {type(result)}")
        
        if isinstance(result, ErrorResponse):
            print(f"❌ Error response: {result.error}")
            raise HTTPException(status_code=400, detail=result.dict())
        
        print(f"✅ Success response with {len(result.forms)} forms")
        return result
        
    except Exception as e:
        print(f"💥 Chat endpoint error: {e}")
        import traceback
        traceback.print_exc()
        
        error_response = ErrorResponse(error=f"Erreur du serveur: {str(e)}")
        raise HTTPException(status_code=500, detail=error_response.dict())

@router.get("/health")
async def chat_health():
    return {"status": "healthy", "service": "RNE Tunisia Chat Pipeline"}

@router.get("/forms")
async def list_forms():
    """List all available RNE forms"""
    try:
        forms = forms_service.get_all_forms()
        return {
            "forms": [form.to_dict() for form in forms],
            "total": len(forms)
        }
    except Exception as e:
        print(f"Error in list_forms: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/forms/test/{query}")
async def test_forms_matching(query: str):
    """Test forms matching for a given query"""
    try:
        forms = forms_service.find_relevant_forms(query)
        return {
            "query": query,
            "matches": [form.to_dict() for form in forms],
            "count": len(forms)
        }
    except Exception as e:
        print(f"Error in test_forms_matching: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})