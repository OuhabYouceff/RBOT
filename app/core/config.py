from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    project_name: str = "RNE Tunisia Chat API"
    version: str = "1.0.0"
    debug: bool = True
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"

settings = Settings()