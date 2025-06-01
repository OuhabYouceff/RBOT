import openai
from openai import OpenAI
from app.core.config import settings
from app.models.chat import ChatRequest, ChatResponse, ErrorResponse
from typing import Union

class ChatService:
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.client = OpenAI(api_key=settings.openai_api_key)
    
    async def get_chat_response(self, chat_request: ChatRequest) -> Union[ChatResponse, ErrorResponse]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": chat_request.message}
                ],
                max_tokens=chat_request.max_tokens,
                temperature=chat_request.temperature
            )
            
            content = response.choices[0].message.content
            return ChatResponse(response=content)
            
        except openai.AuthenticationError:
            return ErrorResponse(error="Invalid OpenAI API key")
        except openai.RateLimitError:
            return ErrorResponse(error="OpenAI API rate limit exceeded")
        except Exception as e:
            return ErrorResponse(error=f"Unexpected error: {str(e)}")

# Create singleton instance
chat_service = ChatService()