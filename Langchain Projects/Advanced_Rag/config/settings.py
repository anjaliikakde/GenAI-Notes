from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings and configurations."""
    
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    langchain_api_key: str = Field(..., env="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(True, env="LANGCHAIN_TRACING_V2")
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # This will ignore extra environment variables

def get_settings():
    """Get application settings."""
    return Settings()