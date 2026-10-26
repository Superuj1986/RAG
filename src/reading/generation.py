
from functools import lru_cache

from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI

from reading.config import get_settings


@lru_cache(maxsize=1)
def configure_llm() -> GoogleGenAI:
    config = get_settings()

    llm = GoogleGenAI(
        model=config.gemini_model,
        api_key=config.get_google_api_key(),
    )

    Settings.llm = llm
    return llm


def generate_answer(query_engine, question: str):
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    return query_engine.query(question)
