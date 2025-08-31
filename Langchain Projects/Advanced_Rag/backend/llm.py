from langchain_openai import ChatOpenAI
from config.settings import get_settings

settings = get_settings()

def get_llm():
    """Initialize and return the language model with configured settings."""
    return ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0,
        api_key=settings.openai_api_key
    )