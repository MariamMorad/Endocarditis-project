"""
Single shared Gemini client. The API key is read from .env via app.config.settings
(no key is ever hardcoded here).
"""
from google import genai
from app.config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)
GEMINI_MODEL = settings.GEMINI_MODEL
