
from functools import lru_cache

from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI

from reading.config import get_settings


@lru_cache(maxsize=4)
def configure_llm(model: str | None = None) -> GoogleGenAI:
    config = get_settings()
    model_name = model or config.gemini_model

    llm = GoogleGenAI(
        model=model_name,
        api_key=config.get_google_api_key(),
    )

    Settings.llm = llm
    return llm


def generate_answer(query_engine, question: str):
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    return query_engine.query(question)


def is_service_unavailable(error: Exception) -> bool:
    message = str(error).lower()
    return "503" in message or "unavailable" in message
