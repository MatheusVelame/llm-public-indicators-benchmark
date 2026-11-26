import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_client(provider: str = "openai") -> OpenAI:
    if provider == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "gemini":
        return OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    elif provider == "groq":
        return OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
    raise ValueError(f"Provedor não suportado: {provider}")