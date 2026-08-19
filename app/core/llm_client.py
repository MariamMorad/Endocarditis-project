"""
Single shared Azure OpenAI client. Configured via app.config.settings.
"""
from openai import OpenAI
from app.config import settings

client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
)
OPENAI_MODEL = settings.OPENAI_MODEL
